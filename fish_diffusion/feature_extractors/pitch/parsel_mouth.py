import parselmouth
import torch
from loguru import logger

from fish_diffusion.utils.tensor import repeat_expand

from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class ParselMouthPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using parselmouth.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        time_step = self.hop_length / sampling_rate

        f0 = (
            parselmouth.Sound(x[0].cpu().numpy(), sampling_rate)
            .to_pitch_ac(
                time_step=time_step,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )

        f0 = torch.from_numpy(f0).to(x.device)

        if pad_to is None:
            return f0

        assert abs(pad_to - len(f0)) < self.hop_length

        return repeat_expand(f0, pad_to)
