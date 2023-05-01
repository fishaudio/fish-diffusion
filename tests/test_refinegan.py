import soundfile as sf
import torchaudio

from fish_diffusion.modules.pitch_extractors import ParselMouthPitchExtractor
from fish_diffusion.modules.vocoders import RefineGAN

source = "dataset/valid/opencpop/TruE-干音_0000/0002.wav"

gan = RefineGAN("checkpoints/refinegan/model", "checkpoints/refinegan/config.json")

audio, sr = torchaudio.load(source)

mel = gan.wav2spec(audio)
from matplotlib import pyplot as plt

plt.imshow(mel.cpu().numpy(), aspect="auto", origin="lower")
plt.savefig("mel_bad.png")
f0 = ParselMouthPitchExtractor(
    f0_min=40.0, f0_max=2000.0, keep_zeros=False, hop_length=gan.config["hop_length"]
)(audio, sr, pad_to=mel.shape[-1])

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
