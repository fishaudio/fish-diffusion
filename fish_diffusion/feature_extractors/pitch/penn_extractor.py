import torch
from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class PennPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using penn.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        import penn

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        x = penn.resample(x, sampling_rate)
        hopsize = self.hop_length / sampling_rate

        f0, _ = penn.from_audio(
            x,
            penn.SAMPLE_RATE,
            hopsize=hopsize,
            fmin=self.f0_min,
            fmax=self.f0_max,
            checkpoint=penn.DEFAULT_CHECKPOINT,
            batch_size=1024,
            pad=False,
            interp_unvoiced_at=0.065,
            gpu=x.device.index,
        )

        f0 = f0[0]

        if pad_to is None:
            return f0

        assert len(f0) <= pad_to and pad_to - len(f0) < self.hop_length

        pad_size = (pad_to - len(f0)) // 2
        f0 = torch.nn.functional.pad(f0, [pad_size, pad_to - len(f0) - pad_size])

        return f0
