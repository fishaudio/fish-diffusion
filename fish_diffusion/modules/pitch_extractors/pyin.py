import librosa
import numpy as np
import resampy

from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class PyinPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using libf0 pyin.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        if sampling_rate != 22050:
            y = resampy.resample(x[0].cpu().numpy(), sampling_rate, 22050)
        else:
            y = x[0].cpu().numpy()

        # Extract pitch using libf0.pyin
        pyin_tuple = librosa.pyin(
            y,
            frame_length=1024,
            fmin=self.f0_min,
            fmax=self.f0_max,
        )

        frequencies = pyin_tuple[0]

        # Replace NaN frames with zeros
        nan_indices = np.isnan(frequencies)
        if np.any(nan_indices):
            frequencies[nan_indices] = 0

        return self.post_process(x, sampling_rate, frequencies, pad_to)
