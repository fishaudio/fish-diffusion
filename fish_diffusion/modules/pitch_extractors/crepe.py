from typing import Literal, Optional

import resampy
import torch
import torchcrepe

from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class CrepePitchExtractor(BasePitchExtractor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = 0.05,
        keep_zeros: bool = False,
        model: Literal["full", "tiny"] = "full",
        use_fast_filters: bool = True,
    ):
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)

        self.threshold = threshold
        self.model = model

        # Use fast filters is already merged into torchcrepe
        # See https://github.com/maxrmorrison/torchcrepe/pull/22

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        if sampling_rate != 16000:
            x0 = resampy.resample(x[0].cpu().numpy(), sampling_rate, 16000)
            x = torch.from_numpy(x0).to(x.device)[None]

        # 重采样后按照 hopsize=80, 也就是 5ms 一帧分析 f0
        f0, pd = torchcrepe.predict(
            x,
            16000,
            80,
            self.f0_min,
            self.f0_max,
            pad=True,
            model=self.model,
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
        )

        # Filter, remove silence, set uv threshold, refer to the original warehouse readme
        pd = torchcrepe.filter.median(pd, 3)
        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, 16000, 80)

        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]

        return self.post_process(x, sampling_rate, f0, pad_to)
