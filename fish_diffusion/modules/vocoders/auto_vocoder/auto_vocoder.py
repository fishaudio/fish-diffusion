import librosa
import pytorch_lightning as pl
import torch
from loguru import logger

from ..builder import VOCODERS
from .models import Decoder, Encoder


@VOCODERS.register_module()
class AutoVocoder(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/auto_vocoder/model.pth",
        encoder=None,
        decoder=None,
        sampling_rate: int = 44100,
    ):
        super().__init__()

        if encoder is None:
            encoder = dict(
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                downsample_rates=[2, 4, 8, 8],
                downsample_kernel_sizes=[4, 8, 16, 16],
                downsample_initial_channel=16,
                hidden_size=128,
            )

        if decoder is None:
            decoder = dict(
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
                upsample_rates=[8, 8, 4, 2],
                upsample_kernel_sizes=[16, 16, 8, 4],
                upsample_initial_channel=512,
                hidden_size=128,
            )

        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)

        cp_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in cp_dict:
            cp_dict = cp_dict["state_dict"]

        # Filter encoder and decoder weights
        cp_dict = {k: v for k, v in cp_dict.items() if "encoder" in k or "decoder" in k}

        self.load_state_dict(cp_dict, strict=True)
        self.eval()

        self.sampling_rate = sampling_rate

    @torch.no_grad()
    def spec2wav(self, spec, f0=None, key_shift=0):
        if key_shift != 0:
            logger.warning("Key shift is not supported for AutoVocoder")

        spec = spec.T
        wav = self.decoder(spec[None])[0]

        return wav

    def wav2spec(self, wav_torch, sr=None, key_shift=0, speed=1.0):
        if sr is None:
            sr = self.sampling_rate

        wav_numpy = wav_torch.cpu().numpy()
        if sr != self.sampling_rate:
            wav_numpy = librosa.resample(
                wav_numpy, orig_sr=sr, target_sr=self.sampling_rate
            )

        if key_shift != 0:
            wav_numpy = librosa.effects.pitch_shift(
                wav_numpy, sr=self.sampling_rate, n_steps=key_shift
            )

        if speed != 1.0:
            wav_numpy = librosa.effects.time_stretch(wav_numpy, speed)

        wav_torch = torch.from_numpy(wav_numpy).to(self.device)
        spec = self.encoder(wav_torch)

        return spec[0].T
