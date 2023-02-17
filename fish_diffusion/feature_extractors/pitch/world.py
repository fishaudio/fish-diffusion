import numpy as np
import pyworld

from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class HarvestPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using world harvest.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        time_step = self.hop_length / sampling_rate * 1000
        double_x = x.cpu().numpy()[0].astype(np.float64)

        f0, t = pyworld.harvest(
            double_x,
            sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=time_step,
        )

        f0 = pyworld.stonemask(double_x, f0, t, sampling_rate)

        return self.post_process(x, sampling_rate, f0, pad_to)


@PITCH_EXTRACTORS.register_module()
class DioPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using world dio.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        time_step = self.hop_length / sampling_rate * 1000
        double_x = x.cpu().numpy()[0].astype(np.float64)

        f0, t = pyworld.dio(
            double_x,
            sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=time_step,
        )

        f0 = pyworld.stonemask(double_x, f0, t, sampling_rate)

        return self.post_process(x, sampling_rate, f0, pad_to)
