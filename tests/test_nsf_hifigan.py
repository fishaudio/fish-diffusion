import soundfile as sf
import torchaudio

from fish_diffusion.modules.pitch_extractors import ParselMouthPitchExtractor
from fish_diffusion.modules.vocoders import NsfHifiGAN

source = "raw/炉心溶解/蹈火-混合.flac"

gan = NsfHifiGAN()

audio, sr = torchaudio.load(source)
audio = audio.mean(dim=0, keepdim=True)

mel = gan.wav2spec(audio)
f0 = ParselMouthPitchExtractor(f0_min=40.0, f0_max=2000.0, keep_zeros=False)(
    audio, sr, pad_to=mel.shape[-1]
)

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
