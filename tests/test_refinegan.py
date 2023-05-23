import soundfile as sf
import torch
import torchaudio
from matplotlib import pyplot as plt

from fish_diffusion.modules.pitch_extractors import (
    CrepePitchExtractor,
    HarvestPitchExtractor,
    ParselMouthPitchExtractor,
)
from fish_diffusion.modules.vocoders import RefineGAN

source = "raw/2008-1.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"

gan = RefineGAN("checkpoints/refinegan/model", "checkpoints/refinegan/config.json")
gan.to(device)
gan.eval()

audio, sr = torchaudio.load(source)
audio = torchaudio.functional.resample(audio, sr, gan.config["sampling_rate"])
audio = audio.mean(0, keepdim=True)
sr = gan.config["sampling_rate"]
audio = audio.to(device)
mel = gan.wav2spec(audio)

f0 = ParselMouthPitchExtractor(
    f0_min=40.0, f0_max=2000.0, keep_zeros=False, hop_length=gan.config["hop_length"]
)(audio, sr, pad_to=mel.shape[-1])

out = gan.spec2wav(mel.to(device), f0.to(device)).cpu().numpy()

sf.write("result.wav", out, sr)
