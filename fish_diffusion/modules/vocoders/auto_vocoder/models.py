import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, BatchNorm2d, Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                BatchNorm2d(out_ch),
                nn.SiLU(),
            ]
        )
        self.out_ch = out_ch
        self.in_ch = in_ch

    def forward(self, x):
        for c in self.convs:
            res = c(x)
            if self.out_ch == self.in_ch:
                x = res + x
            else:
                x = res
        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        *,
        n_blocks,
        latent_dim,
        latent_dropout,
        filter_length,
        hop_length,
        win_length
    ):
        super().__init__()

        self.encs = nn.ModuleList()
        middle = n_blocks // 2 + 1

        for i in range(1, n_blocks + 1):
            if i < middle:
                self.encs.append(ResBlock(4, 4))
            elif i == middle:
                self.encs.append(ResBlock(4, 1))
            else:
                self.encs.append(ResBlock(1, 1))

        self.linear = nn.Linear(win_length // 2 + 1, latent_dim)
        self.dropout = nn.Dropout(latent_dropout)

        # STFT
        self.register_buffer("window", torch.hann_window(win_length))
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, wav):
        # wav: (B, T)
        spec = torch.stft(
            wav,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # (B, N, T, 2)
        mag = torch.sqrt(spec.pow(2).sum(-1).clamp(min=1e-8))  # (B, N, T)
        phase = torch.angle(spec.sum(-1).clamp(min=1e-8))  # (B, N, T)

        x = torch.concat(
            [
                spec,
                mag.unsqueeze(-1),
                phase.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, N, T, 4)

        x = x.permute(0, 3, 1, 2)  # (B, N, T, 4) -> (B, 4, N, T)

        # x: (B, 4, N, T)
        for enc_block in self.encs:
            x = enc_block(x)

        x = x.squeeze(1).transpose(1, 2)  # (B, 1, N, T) -> (B, T, N)
        x = self.linear(x)
        #! Apply dropout (according to DAE) to increase decoder robustness,
        #! because representation predicted from AM is used in TTS application.
        x = self.dropout(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, *, n_blocks, latent_dim, filter_length, hop_length, win_length):
        super().__init__()

        self.linear = nn.Linear(latent_dim, win_length // 2 + 1)
        self.decs = nn.ModuleList()
        middle = n_blocks // 2 + 1

        for i in range(1, n_blocks + 1):
            if i < middle:
                self.decs.append(ResBlock(1, 1))
            elif i == middle:
                self.decs.append(ResBlock(1, 4))
            else:
                self.decs.append(ResBlock(4, 4))

        self.conv_post = Conv2d(4, 2, 3, 1, padding=1)

        # STFT
        self.register_buffer("window", torch.hann_window(win_length))
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x):
        # x: (B, T, D)
        x = self.linear(x)
        x = x.transpose(1, 2).unsqueeze(1)
        for dec_block in self.decs:
            x = dec_block(x)

        # (B, 4, N, T)
        x = F.silu(x)

        # (B, 4, N, T') -> (B, 2, N, T') (default) or (B, 4, N, T')
        x = self.conv_post(x)

        real = x[:, 0, :, :]
        imag = x[:, 1, :, :]

        wav = torch.istft(
            real + 1j * imag,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
        )

        return wav


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.silu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.silu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
