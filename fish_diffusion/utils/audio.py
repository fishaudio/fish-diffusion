import math
from typing import Iterable, Union

import librosa
import numpy as np
import torch
from fish_audio_preprocess.utils import loudness_norm, separate_audio
from torchaudio.transforms import MelSpectrogram


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """

    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """

    return torch.exp(x) / C


def get_mel_transform(
    sample_rate=44100,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    f_min=40,
    f_max=16000,
    n_mels=128,
    center=True,
    power=1.0,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="slaney",
) -> torch.Tensor:
    transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        center=center,
        power=power,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
    )

    return transform


@torch.no_grad()
def get_mel_from_audio(
    audio: torch.Tensor,
    sample_rate=44100,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    f_min=40,
    f_max=16000,
    n_mels=128,
    center=True,
    power=1.0,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="slaney",
) -> torch.Tensor:
    """Get mel spectrogram from audio

    Args:
        audio (torch.Tensor): audio tensor (1, n_samples)

    Returns:
        torch.Tensor: mel spectrogram (n_mels, n_frames)
    """

    assert audio.ndim == 2, "Audio tensor must be 2D (1, n_samples)"
    assert audio.shape[0] == 1, "Audio tensor must be mono"

    transform = get_mel_transform(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        center=center,
        power=power,
        pad_mode=pad_mode,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = transform(audio)
    mel = dynamic_range_compression(mel)

    return mel[0]


def slice_audio(
    audio: np.ndarray,
    rate: int,
    max_duration: float = 30.0,
    top_db: int = 60,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_silence_duration: float = 0,
) -> Iterable[tuple[int, int]]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        max_duration: maximum duration of each slice
        top_db: top_db of librosa.effects.split
        frame_length: frame_length of librosa.effects.split
        hop_length: hop_length of librosa.effects.split
        min_silence_duration: minimum silence duration

    Returns:
        Iterable of start/end frame
    """

    intervals = librosa.effects.split(
        audio.T, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    # Merge intervals that are too close
    if min_silence_duration > 0:
        new_intervals = []
        for start, end in intervals:
            if (
                new_intervals
                and new_intervals[-1][1] + min_silence_duration * rate >= start
            ):
                new_intervals[-1] = (new_intervals[-1][0], end)
            else:
                new_intervals.append((start, end))

        intervals = new_intervals

    for start, end in intervals:
        if end - start <= rate * max_duration:
            # Too short, unlikely to be vocal
            if end - start <= rate * 0.1:
                continue

            yield start, end
            continue

        n_chunks = math.ceil((end - start) / (max_duration * rate))
        chunk_size = math.ceil((end - start) / n_chunks)

        for i in range(start, end, chunk_size):
            yield i, i + chunk_size


def separate_vocals(
    audio: np.ndarray,
    sr: int,
    device: Union[str, torch.device] = "cpu",
    separate_model: torch.nn.Module = None,
):
    if separate_model is not None:
        model = separate_model.to(device)
    else:
        model = separate_audio.init_model("htdemucs", device=device)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=model.samplerate)[None]

    # To two channels
    audio = np.concatenate([audio, audio], axis=0)
    audio = torch.from_numpy(audio).to(device)
    tracks = separate_audio.separate_audio(
        model, audio, shifts=1, num_workers=0, progress=True
    )
    audio = separate_audio.merge_tracks(tracks, filter=["vocals"]).cpu().numpy()
    non_vocals = (
        separate_audio.merge_tracks(tracks, filter=["drums", "bass", "other"])
        .cpu()
        .numpy()
    )

    vocals = librosa.resample(audio[0], orig_sr=model.samplerate, target_sr=sr)
    non_vocals = librosa.resample(non_vocals[0], orig_sr=model.samplerate, target_sr=sr)

    # Normalize loudness
    non_vocals = loudness_norm.loudness_norm(non_vocals, sr)

    return vocals, non_vocals
