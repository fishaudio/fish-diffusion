import numpy as np
import libf0

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

        # Extract pitch using libf0.pyin
        pyin_tuple = libf0.pyin(
                x[0].cpu().numpy(),
                Fs=sampling_rate,
                voicing_prob=0.6,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )

        frequencies = pyin_tuple[0]

        # Replace NaN frames with zeros
        nan_indices = np.isnan(frequencies)
        if np.any(nan_indices):
            frequencies[nan_indices] = 0
        

        f0 = frequencies

        # Pad zeros to the end
        # if pad_to is not None and f0.shape[0] < pad_to:
        #     total_pad = pad_to - f0.shape[0]
        #     f0 = np.pad(f0, (total_pad // 2, total_pad - total_pad // 2), "constant")

        return self.post_process(x, sampling_rate, f0, pad_to)
    
@PITCH_EXTRACTORS.register_module()
class SaliencePitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):

        """Extract pitch using libf0 salience.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        # Extract pitch using libf0.salience
        salience_tuple = libf0.salience(
                x[0].cpu().numpy(),
                Fs=sampling_rate,
                F_min=self.f0_min,
                F_max=self.f0_max,
            )
        
        f0 = salience_tuple[0]

        return self.post_process(x, sampling_rate, f0, pad_to)
    