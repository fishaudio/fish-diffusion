import numpy as np

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
