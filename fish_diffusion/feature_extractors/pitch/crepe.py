import numpy as np
import resampy
import torch
import torchcrepe

from fish_diffusion.utils.tensor import repeat_expand

from .builder import PITCH_EXTRACTORS, BasePitchExtractor


@PITCH_EXTRACTORS.register_module()
class CrepePitchExtractor(BasePitchExtractor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50.0,
        f0_max=1100.0,
        threshold=0.05,
        keep_zeros=False,
    ):
        super().__init__(hop_length, f0_min, f0_max)

        self.threshold = threshold
        self.keep_zeros = keep_zeros

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
            model="full",
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
        )

        # 滤波, 去掉静音, 设置uv阈值, 参考原仓库readme
        pd = torchcrepe.filter.median(pd, 3)
        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, 16000, 80)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

        if self.keep_zeros:
            return repeat_expand(f0[0], pad_to)

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0[0]).squeeze()
        f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
        time_org = 0.005 * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        if f0.shape[0] == 0:
            return torch.zeros(time_frame.shape[0]).float().to(x.device)

        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

        return torch.from_numpy(f0).to(x.device)
