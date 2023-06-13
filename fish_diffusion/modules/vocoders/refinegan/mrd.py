import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class DiscriminatorR(torch.nn.Module):
    def __init__(
        self,
        *,
        n_fft: int = 1024,
        hop_length: int = 120,
        win_length: int = 600,
        use_spectral_norm: bool = False,
        leaky_relu_slope: float = 0.2,
    ):
        super(DiscriminatorR, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.leaky_relu_slope = leaky_relu_slope

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        x = F.pad(
            x,
            (
                int((self.n_fft - self.hop_length) / 2),
                int((self.n_fft - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, resolutions: list[tuple[int]]):
        super(MultiResolutionDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length
                )
                for n_fft, hop_length, win_length in resolutions
            ]
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
