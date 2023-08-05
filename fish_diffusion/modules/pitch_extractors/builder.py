import numpy as np
import torch
from mmengine import Registry
from torch import nn

from fish_diffusion.utils.tensor import interpolate, repeat_expand

PITCH_EXTRACTORS = Registry("pitch_extractors")


class BasePitchExtractor(nn.Module):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        keep_zeros: bool = True,
    ):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """
        super().__init__()

        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError("BasePitchExtractor is not callable.")

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = repeat_expand(f0, pad_to)

        if self.keep_zeros:
            return f0

        # Remove zero frequencies and linearly interpolate
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex)
        time_org = self.hop_length / sampling_rate * nzindex
        time_frame = (
            torch.arange(pad_to, device=x.device) * self.hop_length / sampling_rate
        )

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device)

        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]

        return interpolate(time_frame, time_org, f0, left=f0[0], right=f0[-1])
