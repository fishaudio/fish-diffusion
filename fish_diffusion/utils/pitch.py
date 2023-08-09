import numpy as np
import torch
import torch.nn.functional as F

_f0_bin = 256
_f0_max = 1100.0
_f0_min = 50.0
_f0_mel_min = 1127 * np.log(1 + _f0_min / 700)
_f0_mel_max = 1127 * np.log(1 + _f0_max / 700)


def pitch_to_scale(f0, f0_min=_f0_min, f0_max=_f0_max):
    f0_scale = (f0 - f0_min) / (f0_max - f0_min)

    f0_scale[f0_scale < 0] = 0
    f0_scale[f0_scale > 1] = 1

    # For directly input that only has two dimensions
    if f0.ndim == 2:
        f0_scale = f0_scale.unsqueeze(-1)

    return f0_scale


def pitch_to_mel_scale(
    f0, f0_mel_min=_f0_mel_min, f0_mel_max=_f0_mel_max, f0_bin=_f0_bin
):
    f0_mel = 1127 * (1 + f0 / 700).log()
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    return f0_mel


def pitch_to_coarse(f0, f0_mel_min=_f0_mel_min, f0_mel_max=_f0_mel_max, f0_bin=_f0_bin):
    f0_mel = pitch_to_mel_scale(f0, f0_mel_min, f0_mel_max, f0_bin)
    f0_coarse = (f0_mel + 0.5).long()

    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )

    return f0_coarse


def pitch_to_log(f0):
    x = torch.where(
        f0 > 0,
        f0.log2(),
        torch.zeros_like(f0),
    )

    if x.ndim == 2:
        x = x.unsqueeze(-1)

    return x


def pitch_quant(signals, win_length=16):
    # We want to get the average pitch in each pooling window
    # while ignoring NaNs and zeros.

    assert signals.dim() == 2, "Input tensor must have 2 dimensions (batch_size, width)"
    signals = signals.unsqueeze(1)
    original_shape = signals.shape

    # Need to pad signals to be divisible by win_length
    pad_length = win_length - (signals.shape[-1] % win_length)
    if pad_length != win_length:
        signals = F.pad(signals, (0, pad_length))

    # Apply the mask by setting masked elements to zero, or make NaNs zero
    mask = ~torch.isnan(signals)
    masked_x = torch.where(mask, signals, torch.zeros_like(signals))

    # Create a ones kernel with the same number of channels as the input tensor
    ones_kernel = torch.ones(signals.shape[1], 1, win_length, device=signals.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=win_length,
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=win_length,
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    # Now we have the average pitch in each pooling window.
    # Let's rebuild the original shape.
    expanded = avg_pooled.repeat_interleave(win_length, dim=-1)

    signals = torch.where(
        masked_x != 0,
        expanded,
        masked_x,
    )[:, :, : original_shape[-1]]

    return signals.squeeze(1)


def pitch_to_log_quant(x):
    if x.ndim == 3:
        x = x.squeeze(-1)

    x = pitch_quant(x, 16)

    x = torch.where(
        x > 0,
        x.log2(),
        torch.zeros_like(x),
    )

    if x.ndim == 2:
        x = x.unsqueeze(-1)

    return x
