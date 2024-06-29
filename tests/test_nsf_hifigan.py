import soundfile as sf
import torch
import torchaudio

from fish_diffusion.modules.pitch_extractors import ParselMouthPitchExtractor
from fish_diffusion.modules.vocoders import NsfHifiGAN

source = "raw/2008-1.wav"

gan = NsfHifiGAN("checkpoints/nsf_hifigan/model", "checkpoints/nsf_hifigan/config.json")

audio, sr = torchaudio.load(source)

# Change the multichannel audio to single channel
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)

mel = gan.wav2spec(audio)
f0 = ParselMouthPitchExtractor(f0_min=40.0, f0_max=2000.0, keep_zeros=False)(
    audio, sr, pad_to=mel.shape[-1]
)

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
