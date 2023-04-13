import librosa
import numpy as np
import torch

_f0_bin = 256
_f0_max = 1100.0
_f0_min = 50.0


def get_mel_min_max(f0_min, f0_max):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    return f0_mel_min, f0_mel_max


_f0_mel_min, _f0_mel_max = get_mel_min_max(_f0_min, _f0_max)


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


def pitch_to_mel(f0, f0_mel_min=_f0_mel_min, f0_mel_max=_f0_mel_max, f0_bin=_f0_bin):
    mel_scale = pitch_to_mel_scale(f0, f0_mel_min, f0_mel_max, f0_bin).long()
    mel = torch.nn.functional.one_hot(mel_scale, f0_bin).float().to(f0.device)

    return mel


def mel_scale_to_pitch(
    f0_mel, f0_mel_min=_f0_mel_min, f0_mel_max=_f0_mel_max, f0_bin=_f0_bin
):
    # Clip f0_mel values to the range [1, f0_bin - 1]
    f0_mel = torch.clip(f0_mel, 1, f0_bin - 1)

    # Reverse the mel scaling
    f0_mel = (f0_mel - 1) * (f0_mel_max - f0_mel_min) / (f0_bin - 2) + f0_mel_min

    # Convert mel frequency back to Hz
    f0 = 700 * (torch.exp(f0_mel / 1127) - 1)

    return f0


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


def viterbi_decode(probs):
    xx, yy = np.meshgrid(range(128), range(128))
    transition = np.maximum(4 - abs(xx - yy), 0)
    transition = transition / transition.sum(axis=1, keepdims=True)

    use_torch = False
    sequences = probs
    if isinstance(sequences, torch.Tensor):
        use_torch = True
        sequences = sequences.detach().cpu().numpy()

    decoded = librosa.sequence.viterbi(sequences, transition).astype(np.int64)

    if use_torch:
        decoded = torch.from_numpy(decoded).to(probs.device)

    return decoded
