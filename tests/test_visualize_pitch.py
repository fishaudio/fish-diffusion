import numpy as np
import torchaudio
from librosa.core import hz_to_mel
from loguru import logger
from matplotlib import pyplot as plt

from fish_diffusion.feature_extractors.pitch import (
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

audio, sr = torchaudio.load("raw/孤勇者_Unnamed_Track_1.wav")
mel = get_mel_from_audio(audio, sr, f_min=f_min, f_max=f_max, n_mels=n_mels).numpy()
logger.info(f"Got mel spectrogram with shape {mel.shape}")

parselmouth_pitch_extractor = ParselMouthPitchExtractor(use_repeat_expand=True)
parselmouth_pitch = parselmouth_pitch_extractor(audio, sr, pad_to=mel.shape[-1])
parselmouth_pitch = f_to_mel(parselmouth_pitch)
parselmouth_pitch[parselmouth_pitch <= 0] = float("nan")
logger.info(f"Got parselmouth pitch with shape {parselmouth_pitch.shape}")

penn_pitch_extractor = PennPitchExtractor()
penn_pitch = penn_pitch_extractor(audio, sr, pad_to=mel.shape[-1])
penn_pitch = f_to_mel(penn_pitch)
penn_pitch[penn_pitch <= 0] = float("nan")
logger.info(f"Got penn pitch with shape {penn_pitch.shape}")

harvest_pitch_extractor = HarvestPitchExtractor()
harvest_pitch = harvest_pitch_extractor(audio, sr, pad_to=mel.shape[-1])
harvest_pitch = f_to_mel(harvest_pitch)
harvest_pitch[harvest_pitch <= 0] = float("nan")
logger.info(f"Got harvest pitch with shape {harvest_pitch.shape}")


fig, axs = plt.subplots(3, 1, figsize=(10, 6))
fig.suptitle("Pitch on mel spectrogram")

axs[0].set_title("parselmouth")
axs[0].imshow(mel, aspect="auto", origin="lower")
axs[0].plot(parselmouth_pitch, label="parselmouth", color="red")

axs[1].set_title("penn")
axs[1].imshow(mel, aspect="auto", origin="lower")
axs[1].plot(penn_pitch, label="penn", color="red")

axs[2].set_title("harvest")
axs[2].imshow(mel, aspect="auto", origin="lower")
axs[2].plot(harvest_pitch, label="harvest", color="red")

plt.show()
