import numpy as np
import parselmouth
import resampy
import torch
import torchcrepe
from mmengine import Registry
import pyworld as world

PITCH_EXTRACTORS = Registry("pitch_extractors")

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


def get_pitch_parselmouth(
    x,
    sampling_rate=44100,
    hop_length=512,
    f0_min=_f0_min,
    f0_max=_f0_max,
    pad_to=None,
):
    """Extract pitch using parselmouth.

    Args:
        x (torch.Tensor): Audio signal, shape (1, T).
        sampling_rate (int, optional): Sampling rate. Defaults to 44100.
        hop_length (int, optional): Hop length. Defaults to 512.
        f0_min (float, optional): Minimum pitch. Defaults to 50.
        f0_max (float, optional): Maximum pitch. Defaults to 1100.
        pad_to (int, optional): Pad to length. Defaults to None.

    Returns:
        torch.Tensor: Pitch, shape (T // hop_length,).
    """

    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
    assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

    time_step = hop_length / sampling_rate

    f0 = (
        parselmouth.Sound(x[0].cpu().numpy(), sampling_rate)
        .to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    f0 = torch.from_numpy(f0).to(x.device)

    if pad_to is None:
        return f0

    assert len(f0) <= pad_to and pad_to - len(f0) < hop_length

    pad_size = (pad_to - len(f0)) // 2
    f0 = torch.nn.functional.pad(f0, [pad_size, pad_to - len(f0) - pad_size])

    return f0


def get_pitch_crepe(
    x,
    sampling_rate=44100,
    hop_length=512,
    f0_min=_f0_min,
    f0_max=_f0_max,
    threshold=0.05,
    pad_to=None,
):
    """Extract pitch using crepe.

    Args:
        x (torch.Tensor): Audio signal, shape (1, T).
        sampling_rate (int, optional): Sampling rate. Defaults to 44100.
        hop_length (int, optional): Hop length. Defaults to 512.
        f0_min (float, optional): Minimum pitch. Defaults to 50.
        f0_max (float, optional): Maximum pitch. Defaults to 1100.
        threshold (float, optional): Threshold for unvoiced. Defaults to 0.05.
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
        f0_min,
        f0_max,
        pad=True,
        model="full",
        batch_size=1024,
        device=x.device,
        return_periodicity=True,
    )

    # 滤波, 去掉静音, 设置uv阈值, 参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.0)(pd, x, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # 去掉0频率, 并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(pad_to) * hop_length / sampling_rate

    if f0.shape[0] == 0:
        return torch.zeros(time_frame.shape[0]).float().to(x.device)

    # 大概可以用 torch 重写?
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

    return torch.from_numpy(f0).to(x.device)

def get_pitch_world(
    x,
    sampling_rate=44100,
    hop_length=512,
    f0_min= 50,
    f0_max= 1100,
):
    """Extract pitch using world harvest.

    Args:
        sampling_rate (int, optional): Sampling rate. Defaults to 44100.
        hop_length (int, optional): Hop length. Defaults to 512.
        f0_min (float, optional): Minimum pitch. Defaults to 50.
        f0_max (float, optional): Maximum pitch. Defaults to 1100.
    """

    #I tried to add a pad handler in case it isnt none, i couldn't
    time_step = hop_length / sampling_rate
    f0, _ = world.harvest(sampling_rate, f0_min, f0_max, time_step)
    return f0


class BasePitchExtractor:
    def __init__(self, hop_length=512, f0_min=_f0_min, f0_max=_f0_max):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError


@PITCH_EXTRACTORS.register_module()
class ParselMouthPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        return get_pitch_parselmouth(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            pad_to,
        )


@PITCH_EXTRACTORS.register_module()
class CrepePitchExtractor(BasePitchExtractor):
    def __init__(self, hop_length=512, f0_min=_f0_min, f0_max=_f0_max, threshold=0.05):
        super().__init__(hop_length, f0_min, f0_max)
        self.threshold = threshold

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        return get_pitch_crepe(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            self.threshold,
            pad_to,
        )

@PITCH_EXTRACTORS.register_module()
class HarvestMouthPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100):
        return get_pitch_world(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
        )
