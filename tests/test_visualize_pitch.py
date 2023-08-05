import librosa
import numpy as np
import torch
from librosa.core import hz_to_mel, mel_frequencies
from loguru import logger
from matplotlib import pyplot as plt

from fish_diffusion.modules.pitch_extractors import (
    CrepePitchExtractor,
    DioPitchExtractor,
    HarvestPitchExtractor,
    ParselMouthPitchExtractor,
    RMVPitchExtractor,
)
from fish_diffusion.utils.audio import get_mel_from_audio

device = "cuda" if torch.cuda.is_available() else "cpu"

f_min = 40
f_max = 16000
n_mels = 128

min_mel = hz_to_mel(f_min)
max_mel = hz_to_mel(f_max)
f_to_mel = lambda x: (hz_to_mel(x) - min_mel) / (max_mel - min_mel) * n_mels
mel_freqs = mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)

audio, sr = librosa.load(
    "dataset/train/aria/炫境干音_0000/0022.wav",
    sr=44100,
    mono=True,
)
audio = torch.from_numpy(audio).unsqueeze(0).to(device)

mel = (
    get_mel_from_audio(audio, sr, f_min=f_min, f_max=f_max, n_mels=n_mels).cpu().numpy()
)
logger.info(f"Got mel spectrogram with shape {mel.shape}")

extractors = {
    "RMVPE": RMVPitchExtractor,
    "Crepe": CrepePitchExtractor,
    "ParselMouth": ParselMouthPitchExtractor,
    "Harvest": HarvestPitchExtractor,
    "Dio": DioPitchExtractor,
}

fig, axs = plt.subplots(len(extractors), 1, figsize=(10, len(extractors) * 3))
fig.suptitle("Pitch on mel spectrogram")

for idx, (name, extractor) in enumerate(extractors.items()):
    extra_kwargs = {
        "keep_zeros": False,
    }

    pitch_extractor = extractor(f0_min=40.0, f0_max=1600, **extra_kwargs).to(device)
    f0 = pitch_extractor(audio, sr, pad_to=mel.shape[-1]).cpu().numpy()
    f0 = f_to_mel(f0)
    f0[f0 <= 0] = float("nan")
    logger.info(f"Got {name} pitch with shape {f0.shape}")

    ax = axs[idx]
    ax.set_title(name)
    ax.imshow(mel, aspect="auto", origin="lower")
    ax.plot(f0, label=name, color="red")
    ax.set_yticks(np.arange(0, n_mels, 10))
    ax.set_yticklabels(np.round(mel_freqs[::10]).astype(int))
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (frames)")
    ax.legend()

plt.savefig("pitch.png")
plt.show()
