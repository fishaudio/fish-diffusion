import numpy as np
import parselmouth
import torch
import torchcrepe
import torchaudio
import librosa

_f0_bin = 256
_f0_max = 1100.0
_f0_min = 50.0
_f0_mel_min = 1127 * np.log(1 + _f0_min / 700)
_f0_mel_max = 1127 * np.log(1 + _f0_max / 700)


def f0_to_coarse(f0, f0_mel_min=_f0_mel_min, f0_mel_max=_f0_mel_max, f0_bin=_f0_bin):
    is_torch = isinstance(f0, torch.Tensor)

    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)

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
    mel=None,
):
    """Extract pitch using parselmouth.

    Args:
        x (torch.Tensor): Audio signal, shape (T,).
        sampling_rate (int, optional): Sampling rate. Defaults to 44100.
        hop_length (int, optional): Hop length. Defaults to 512.
        f0_min (float, optional): Minimum pitch. Defaults to 50.
        f0_max (float, optional): Maximum pitch. Defaults to 1100.

    Returns:
        torch.Tensor: Pitch, shape (T // hop_length,).
    """

    x = x.cpu().numpy()
    time_step = hop_length / sampling_rate

    f0 = (
        parselmouth.Sound(x, sampling_rate)
        .to_pitch_ac(
            time_step=time_step,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    p_len = x.shape[0] // hop_length
    pad_size = (p_len - len(f0) + 1) // 2

    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

    return torch.from_numpy(f0)


def get_pitch_crepe(
    x,
    mel,
    sampling_rate=44100,
    hop_length=512,
    f0_min=_f0_min,
    f0_max=_f0_max,
    threshold=0.05,
):
    """Extract pitch using crepe.

    Args:
        x (torch.Tensor): Audio signal, shape (T,).
        sampling_rate (int, optional): Sampling rate. Defaults to 44100.
        hop_length (int, optional): Hop length. Defaults to 512.
        f0_min (float, optional): Minimum pitch. Defaults to 50.
        f0_max (float, optional): Maximum pitch. Defaults to 1100.
        threshold (float, optional): Threshold for unvoiced. Defaults to 0.05.

    Returns:
        torch.Tensor: Pitch, shape (T // hop_length,).
    """

    if sampling_rate != 16000:
        x = torchaudio.functional.resample(x, sampling_rate, 16000)

    # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    f0, pd = torchcrepe.predict(
        x,
        16000,
        80,
        f0_min,
        f0_max,
        pad=True, 
        model='full', 
        batch_size=1024,
        device=x.device, 
        return_periodicity=True
    )

    # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, x, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)
    
    # 去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(len(mel)) * hop_length / sampling_rate

    if f0.shape[0] == 0:
        return torch.FloatTensor(time_frame.shape[0]).fill_(0)

    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

    return torch.from_numpy(f0)
