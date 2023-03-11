from typing import Literal

import librosa
import torch
from torch import nn

from fish_diffusion.utils.tensor import repeat_expand

from .builder import ENERGY_EXTRACTORS


@ENERGY_EXTRACTORS.register_module()
class RMSEnergyExtractor(nn.Module):
    def __init__(
        self,
        frame_length=2048,
        hop_length=512,
        center=True,
        pad_mode: Literal["reflect", "constant", "edge", "wrap"] = "reflect",
    ):
        super().__init__()

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract energy from audio using librosa.feature.rms.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Energy, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        device = x.device
        x = x.squeeze(0).cpu().numpy()

        energy = librosa.feature.rms(
            y=x,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=self.center,
            pad_mode=self.pad_mode,
        )

        energy = torch.from_numpy(energy).squeeze(-2).to(device)

        if pad_to is None:
            return energy

        return repeat_expand(energy, pad_to)
