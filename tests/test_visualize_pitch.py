import numpy as np
import torchaudio
from librosa.core import hz_to_mel
from loguru import logger
from matplotlib import pyplot as plt

from fish_diffusion.feature_extractors.pitch import (
    CrepePitchExtractor,
    HarvestPitchExtractor,
    ParselMouthPitchExtractor,
    PennPitchExtractor,
)
from fish_diffusion.utils.audio import get_mel_from_audio

f_min = 40
f_max = 16000
n_mels = 128

min_mel = hz_to_mel(f_min)
max_mel = hz_to_mel(f_max)
f_to_mel = lambda x: (hz_to_mel(x) - min_mel) / (max_mel - min_mel) * n_mels

audio, sr = torchaudio.load("raw/炼丹组/干声.wav")
mel = get_mel_from_audio(audio, sr, f_min=f_min, f_max=f_max, n_mels=n_mels).numpy()
logger.info(f"Got mel spectrogram with shape {mel.shape}")

extractors = {
    "Crepe": CrepePitchExtractor,
    "ParselMouth": ParselMouthPitchExtractor,
    "Penn": PennPitchExtractor,
    "Harvest": HarvestPitchExtractor,
}

fig, axs = plt.subplots(len(extractors), 1, figsize=(10, len(extractors) * 3))
fig.suptitle("Pitch on mel spectrogram")

for idx, (name, extractor) in enumerate(extractors.items()):
    pitch_extractor = extractor()
    f0 = pitch_extractor(audio, sr, pad_to=mel.shape[-1])
    f0 = f_to_mel(f0)
    f0[f0 <= 0] = float("nan")
    logger.info(f"Got {name} pitch with shape {f0.shape}")

    ax = axs[idx]
    ax.set_title(name)
    ax.imshow(mel, aspect="auto", origin="lower")
    ax.plot(f0, label=name, color="red")
    ax.legend()

plt.savefig("pitch.png")
plt.show()
