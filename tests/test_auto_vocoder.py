import soundfile as sf
import torchaudio

from fish_diffusion.modules.vocoders import AutoVocoder

vocoder = AutoVocoder(
    "logs/AutoVocoder/version_None/checkpoints/epoch=258-step=270000-valid_loss=0.25.ckpt"
)

source = "raw/炉心溶解/蹈火-混合.flac"
audio, sr = torchaudio.load(source)

spec = vocoder.wav2spec(audio)
out = vocoder.spec2wav(spec)

sf.write("result.wav", out, sr)
