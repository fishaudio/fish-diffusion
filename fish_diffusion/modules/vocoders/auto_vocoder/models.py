import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

from fish_diffusion.modules.vocoders.nsf_hifigan.models import get_padding


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()

        self.m = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=get_padding(kernel_size, dilation[i]),
                    ),
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

        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        *,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        downsample_rates,
        downsample_kernel_sizes,
        downsample_initial_channel,
        hidden_size,
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

        self.conv_post_mu = Conv1d(ch, hidden_size, 7, 1, padding=3)
        self.conv_post_sigma = Conv1d(ch, hidden_size, 7, 1, padding=3)

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

        mu = self.conv_post_mu(x)
        sigma = torch.exp(self.conv_post_sigma(x))

        return mu, sigma

    def sample(self, mu, sigma):
        return mu + sigma * torch.randn_like(mu)


class Decoder(torch.nn.Module):
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


if __name__ == "__main__":

    def count_parameters(model):
        x = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return f"{x / 1e6:.2f}M"

    decoder = Decoder(
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        upsample_rates=[8, 8, 4, 2],
        upsample_kernel_sizes=[16, 16, 8, 4],
        upsample_initial_channel=512,
        hidden_size=128,
    )

    print(count_parameters(decoder))

    encoder = Encoder(
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        downsample_rates=[2, 4, 8, 8],
        downsample_kernel_sizes=[4, 8, 16, 16],
        downsample_initial_channel=16,
        hidden_size=128,
    )

    print(count_parameters(encoder))

    x = torch.randn(1, 1, 16384 * 5)
    y = encoder(x)
    print(y.shape)

    z = decoder(y)
    print(z.shape)

    assert z.shape == x.shape
