import soundfile as sf
import torchaudio

from fish_diffusion.modules.pitch_extractors import ParselMouthPitchExtractor
from fish_diffusion.modules.vocoders import RefineGAN
from tools.refinegan.train import RefineGAN as RefineGANL

source = "dataset/valid/opencpop/TruE-干音_0000/0002.wav"

gan = RefineGAN("checkpoints/refinegan/model", "checkpoints/refinegan/config.json")
from mmengine import Config
config = Config.fromfile("configs/vocoder_refinegan.py")
m = RefineGANL.load_from_checkpoint("logs/RefineGAN/jcx5oknz/checkpoints/epoch=29-step=80000-valid_loss=0.3222.ckpt", config=config)
audio, sr = torchaudio.load(source)
gan.generator = m.generator
gan.generator.eval()
gan.generator.remove_weight_norm()
mel = gan.wav2spec(audio)
f0 = ParselMouthPitchExtractor(f0_min=40.0, f0_max=2000.0, keep_zeros=False)(
    audio, sr, pad_to=mel.shape[-1]
)

out = gan.spec2wav(mel, f0)

sf.write("result.wav", out, sr)
