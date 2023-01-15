import numpy as np
import soundfile as sf
import torch
import torchaudio

from fish_diffusion.vocoders import NsfHifiGAN

source = "INPUT.wav"
f0 = np.load(source + ".f0.npy")
f0 = torch.from_numpy(f0).float()

gan = NsfHifiGAN()

audio, sr = torchaudio.load(source)
mel = gan.wav2spec(audio)
a = gan.spec2wav(mel, f0)

sf.write("result.wav", a, 44100)
