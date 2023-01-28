import soundfile as sf
import torchaudio

from fish_diffusion.utils.pitch import get_pitch_parselmouth
from fish_diffusion.vocoders import NsfHifiGAN

source = "dataset/valid-for-opencpop/TruE-干音_0000/0000.wav"

gan = NsfHifiGAN()

audio, sr = torchaudio.load(source)

mel = gan.wav2spec(audio)
f0 = get_pitch_parselmouth(audio, sr, pad_to=mel.shape[-1])

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
