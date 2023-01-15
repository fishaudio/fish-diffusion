import numpy as np
import librosa
from fish_diffusion.utils.pitch import f0_to_coarse, get_pitch_parselmouth
from fish_diffusion.utils.audio import get_mel_from_audio

path = "data/diff-svc-clean/aria/All of me 干音/0000.wav.soft.npy"

c = np.load(path)

print(c.shape)

"tools/preprocess/test.py"


def process(audio_path):
    audio = librosa.load(audio_path, sr=16000, mono=True)[0]
