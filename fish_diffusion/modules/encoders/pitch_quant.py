import torch
import torch.nn.functional as F
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class QuantizedPitchEncoder(nn.Module):
    def __init__(self, output_size, win_length=16):
        super().__init__()

        bins = 440 * 2 ** ((torch.arange(96) - 48) / 12)
        self.register_buffer("bins", bins)

        self.win_length = win_length
        self.embedding = nn.Embedding(len(bins), output_size)

    def forward(self, signals):
        if signals.ndim == 3 and signals.shape[-1] == 1:
            signals = signals.squeeze(-1)

        assert (
            signals.dim() == 2
        ), "Input tensor must have 2 dimensions (batch_size, width)"
        signals = signals.unsqueeze(1)
        original_shape = signals.shape

        # Need to pad signals to be divisible by win_length
        pad_length = self.win_length - (signals.shape[-1] % self.win_length)
        if pad_length != self.win_length:
            signals = F.pad(signals, (0, pad_length))

        # Apply the mask by setting masked elements to zero, or make NaNs zero
        mask = ~torch.isnan(signals)
        masked_x = torch.where(mask, signals, torch.zeros_like(signals))

        # Create a ones kernel with the same number of channels as the input tensor
        ones_kernel = torch.ones(
            signals.shape[1], 1, self.win_length, device=signals.device
        )

        # Perform sum pooling
        sum_pooled = F.conv1d(
            masked_x,
            ones_kernel,
            stride=self.win_length,
        )

        # Count the non-masked (valid) elements in each pooling window
        valid_count = F.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.win_length,
        )
        valid_count = valid_count.clamp(min=1)  # Avoid division by zero

        # Perform masked average pooling
        avg_pooled = sum_pooled / valid_count

        # Now we have the average pitch in each pooling window.
        # Let's rebuild the original shape.
        expanded = avg_pooled.repeat_interleave(self.win_length, dim=-1)

        signals = torch.where(
            masked_x != 0,
            expanded,
            masked_x,
        )[:, :, : original_shape[-1]]

        x = signals.squeeze(1)

        # Quantize to closest pitch
        # Bins shape: (96,), x shape: (batch_size, width)
        x = torch.abs(x.unsqueeze(-1) - self.bins).argmin(dim=-1)
        x = self.embedding(x)

        return x
