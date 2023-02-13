import soundfile as sf
import torchaudio

from fish_diffusion.feature_extractors.pitch import ParselMouthPitchExtractor
from fish_diffusion.vocoders import NsfHifiGAN

source = "dataset/valid-for-opencpop/TruE-干音_0000/0000.wav"

gan = NsfHifiGAN()

audio, sr = torchaudio.load(source)

mel = gan.wav2spec(audio)
f0 = ParselMouthPitchExtractor(
    f0_min=40.0,
    f0_max=1600.0,
)(audio, sr, pad_to=mel.shape[-1])

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
