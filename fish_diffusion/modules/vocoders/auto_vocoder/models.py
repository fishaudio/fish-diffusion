import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, BatchNorm1d

from fish_diffusion.modules.vocoders.nsf_hifigan.models import get_padding
from encodec.modules.seanet import SEANetEncoder, SEANetDecoder

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()

        self.m = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(),
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                )
                for i in range(3)
            ]
        )

    def forward(self, x):
        for m in self.m:
            x = m(x) + x
            x = F.silu(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        downsample_rates,
        downsample_kernel_sizes,
        downsample_initial_channel,
        hidden_size,
        double_z=True,
    ):
        super(Encoder, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_downsamples = len(downsample_rates)
        self.conv_pre = Conv1d(1, downsample_initial_channel, 7, 1, padding=3)

        self.downs = nn.ModuleList()
        for i, (d, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
            self.downs.append(
                Conv1d(
                    downsample_initial_channel * (2**i),
                    downsample_initial_channel * (2 ** (i + 1)),
                    k,
                    d,
                    padding=(k - d) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = downsample_initial_channel * (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # self.conv_post = nn.Sequential(
        #     nn.SiLU(),
        #     Conv1d(ch, hidden_size * 2 if double_z else hidden_size, 7, 1, padding=3),
        # )
        self.conv_post = Conv1d(ch, hidden_size, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.silu(x)
            x = self.downs[i](x)
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels
        
        x = F.silu(x)

        return self.conv_post(x)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        hidden_size,
    ):
        super(Decoder, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(hidden_size, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.silu(x)
            x = self.ups[i](x)
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = F.silu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class AutoEncoderKL(nn.Module):
    def __init__(self, *, encoder=None, decoder=None, hidden_size, embedding_size):
        super(AutoEncoderKL, self).__init__()

        self.encoder = SEANetEncoder(**encoder)
        self.decoder = SEANetDecoder(**decoder)

        self.quant_conv = nn.Conv1d(hidden_size * 2, embedding_size * 2, 1)
        self.post_quant_conv = nn.Conv1d(embedding_size, hidden_size, 1)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)

        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        return mean, logvar

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)

        return x

    def forward(self, x, with_loss=True, deterministic=False):
        mean, logvar = self.encode(x)

        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)

        if deterministic is False:
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        x = self.decode(z)

        if with_loss is False:
            return x

        # Only sum on channel dimension
        loss = 0.5 * torch.sum(mean**2 + var - 1.0 - logvar, dim=[1]).mean()

        return x, loss
