import soundfile as sf
import torchaudio

from fish_diffusion.modules.vocoders.auto_vocoder.auto_vocoder import AutoVocoder

vocoder = AutoVocoder(
    "logs/AutoVocoder/version_None/checkpoints/epoch=152-step=160000-valid_loss=0.29.ckpt"
)


source = "dataset/valid/opencpop/TruE-干音_0000/0002.wav"
audio, sr = torchaudio.load(source)

spec = vocoder.wav2spec(audio)
out = vocoder.spec2wav(spec)

sf.write("result.wav", out, sr)
