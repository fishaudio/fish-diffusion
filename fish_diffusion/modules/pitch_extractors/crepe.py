from typing import Literal, Optional

import resampy
import torch
import torchcrepe
from torch import nn
from torch.nn import functional as F

from fish_diffusion.modules.pitch_extractors.builder import BasePitchExtractor


class MaskedAvgPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # Apply the mask by setting masked elements to zero, or make NaNs zero
        if mask is None:
            mask = ~torch.isnan(x)

        # Ensure mask has the same shape as the input tensor
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))
        # Create a ones kernel with the same number of channels as the input tensor
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)

        # Perform sum pooling
        sum_pooled = nn.functional.conv1d(
            masked_x,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )

        # Count the non-masked (valid) elements in each pooling window
        valid_count = nn.functional.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )
        valid_count = valid_count.clamp(min=1)  # Avoid division by zero

        # Perform masked average pooling
        avg_pooled = sum_pooled / valid_count

        # Fill zero values with NaNs
        avg_pooled[avg_pooled == 0] = float("nan")

        if ndim == 2:
            return avg_pooled.squeeze(1)

        return avg_pooled


class MaskedMedianPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)

        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        if mask is None:
            mask = ~torch.isnan(x)

        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        masked_x = torch.where(mask, x, torch.zeros_like(x))

        x = F.pad(masked_x, (self.padding, self.padding), mode="reflect")
        mask = F.pad(
            mask.float(), (self.padding, self.padding), mode="constant", value=0
        )

        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)

        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,))

        # Combine the mask with the input tensor
        x_masked = torch.where(mask.bool(), x, float("inf"))

        # Sort the masked tensor along the last dimension
        x_sorted, _ = torch.sort(x_masked, dim=-1)

        # Compute the count of non-masked (valid) values
        valid_count = mask.sum(dim=-1)

        # Calculate the index of the median value for each pooling window
        median_idx = ((valid_count - 1) // 2).clamp(min=0)

        # Gather the median values using the calculated indices
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

        # Fill infinite values with NaNs
        median_pooled[torch.isinf(median_pooled)] = float("nan")

        if ndim == 2:
            return median_pooled.squeeze(1)

        return median_pooled


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
        self.use_fast_filters = use_fast_filters

        if self.use_fast_filters:
            self.median_filter = MaskedMedianPool1d(3, 1, 1)
            self.mean_filter = MaskedAvgPool1d(3, 1, 1)

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
        if self.use_fast_filters:
            pd = self.median_filter(pd)
        else:
            pd = torchcrepe.filter.median(pd, 3)

        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, 16000, 80)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)

        if self.use_fast_filters:
            f0 = self.mean_filter(f0)
        else:
            f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]

        return self.post_process(x, sampling_rate, f0, pad_to)
