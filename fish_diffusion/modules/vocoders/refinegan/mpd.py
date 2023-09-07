from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        *,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        leaky_relu_slope: float = 0.2,
        channels: Optional[list[int]] = None,
    ) -> None:
        super(DiscriminatorP, self).__init__()

        self.period = period
        self.leaky_relu_slope = leaky_relu_slope
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        if channels is None:
            channels = [1, 64, 128, 256, 512, 1024]

        channel = 1
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))

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
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Optional[list[int]] = None):
        super(MultiPeriodDiscriminator, self).__init__()

        if periods is None:
            periods = [2, 3, 5, 7, 11]

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=period) for period in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        ret_x, ret_fmap = [], []

        for disc in self.discriminators:
            res, fmap = disc(x)

            ret_x.append(res)
            ret_fmap.append(fmap)

        return ret_x, ret_fmap
