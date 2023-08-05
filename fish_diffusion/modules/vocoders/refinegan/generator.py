from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super(ResBlock, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels if idx == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs1.apply(self.init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for idx, d in enumerate(dilation)
            ]
        )
        self.convs2.apply(self.init_weights)

    def forward(self, x):
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)

            if idx != 0 or self.in_channels == self.out_channels:
                x = xt + x
            else:
                x = xt

        return x

    def remove_weight_norm(self):
        for c1, c2 in zip(self.convs1, self.convs2):
            remove_weight_norm(c1)
            remove_weight_norm(c2)

    def init_weights(self, m):
        if type(m) == nn.Conv1d:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.0)


class AdaIN(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: int = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(channels=out_channels),
                    ResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(channels=out_channels),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        results = [block(x) for block in self.blocks]

        return torch.mean(torch.stack(results), dim=0)

    def remove_weight_norm(self):
        for block in self.blocks:
            block[1].remove_weight_norm()


class CombToothGen(nn.Module):
    def __init__(
        self,
        *,
        sampling_rate: int = 44100,
        wave_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.wave_amp = wave_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0 (torch.Tensor): [B, 1, T]

        Returns:
            combtooth (torch.Tensor): [B, 1, T]
        """

        x = torch.cumsum(f0 / self.sampling_rate, axis=2)
        x = x - torch.round(x)
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3)) * self.wave_amp

        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.noise_std + (1 - uv) * self.wave_amp / 3
        noise = noise_amp * torch.randn_like(combtooth)
        combtooth = combtooth * uv + noise

        return combtooth


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGen, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        # To prevent torch.cumsum numerical overflow,
        # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
        # Buffer tmp_over_one_idx indicates the time step to add -1.
        # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        # Clean up overflows
        sines[f0_values > self.sampling_rate // 2] = 0

        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, 1)
        """
        f0 = f0.transpose(1, 2)

        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)

            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise

        # Need grad before merge
        return self.merge(sine_waves).transpose(1, 2)


class RefineGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        sampling_rate: int = 44100,
        hop_length: int = 256,
        downsample_rates: tuple[int] = (2, 2, 8, 8),
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        template_generator: Literal["comb", "sine"] = "comb",
    ) -> None:
        super().__init__()

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.downsample_rates = downsample_rates
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope

        assert np.prod(downsample_rates) == np.prod(upsample_rates) == hop_length

        if template_generator == "comb":
            self.template_gen = CombToothGen(sampling_rate=sampling_rate)
        elif template_generator == "sine":
            self.template_gen = SineGen(sampling_rate=sampling_rate)
        else:
            raise ValueError(f"Unknown template generator: {template_generator}")

        self.template_conv = weight_norm(
            nn.Conv1d(
                in_channels=1,
                out_channels=start_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        channels = start_channels

        self.downsample_blocks = nn.ModuleList([])
        for rate in downsample_rates:
            new_channels = channels * 2

            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=1 / rate, mode="linear"),
                    ResBlock(
                        in_channels=channels,
                        out_channels=new_channels,
                        kernel_size=7,
                        dilation=(1, 3, 5),
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                )
            )

            channels = new_channels

        self.mel_conv = weight_norm(
            nn.Conv1d(
                in_channels=num_mels,
                out_channels=channels,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        channels *= 2

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        stride_f0 = np.prod(upsample_rates[1:])
        self.source_conv = nn.Conv1d(
            1,
            channels,
            kernel_size=stride_f0 * 2,
            stride=stride_f0,
            padding=stride_f0 // 2,
        )

        for rate in upsample_rates:
            new_channels = channels // 2

            self.upsample_blocks.append(nn.Upsample(scale_factor=rate, mode="linear"))
            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.output_conv = weight_norm(
            nn.Conv1d(
                in_channels=channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.template_conv)
        remove_weight_norm(self.mel_conv)
        remove_weight_norm(self.output_conv)

        for block in self.downsample_blocks:
            block[1].remove_weight_norm()

        for block in self.upsample_conv_blocks:
            block.remove_weight_norm()

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (torch.Tensor): [B, mel_bin, T]
            f0 (torch.Tensor): [B, 1, T]

        Returns:
            combtooth (torch.Tensor): [B, 1, T * hop_length]
        """

        f0 = F.interpolate(f0, size=mel.shape[-1] * self.hop_length, mode="linear")
        template = self.template_gen(f0)

        x = self.template_conv(template)

        downs = []

        for block in self.downsample_blocks:
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = block(x)

        x = torch.cat([x, self.mel_conv(mel)], dim=1)

        for idx, (upsample_block, conv_block, down) in enumerate(
            zip(
                self.upsample_blocks,
                self.upsample_conv_blocks,
                reversed(downs),
            )
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = upsample_block(x)

            if idx == 0:
                x = x + self.source_conv(template)

            x = torch.cat([x, down], dim=1)
            x = conv_block(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.output_conv(x)
        x = torch.tanh(x)

        return x
