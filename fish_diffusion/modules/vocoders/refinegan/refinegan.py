import json
from pathlib import Path
from typing import Optional

import librosa
import pytorch_lightning as pl
import torch

from fish_diffusion.utils.audio import dynamic_range_compression, get_mel_transform
from fish_diffusion.utils.pitch_adjustable_mel import PitchAdjustableMelSpectrogram

from ..builder import VOCODERS
from .generator import RefineGANGenerator


@VOCODERS.register_module()
class RefineGAN(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/refinegan/model",
        config_file: Optional[str] = None,
        use_natural_log: bool = True,
    ):
        super().__init__()

        if config_file is None:
            config_file = Path(checkpoint_path).parent / "config.json"

        with open(config_file) as f:
            data = f.read()

        config = json.loads(data)
        self.model = RefineGANGenerator(**config["generator"])
        self.use_natural_log = use_natural_log
        self.config = config

        cp_dict = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" not in cp_dict:
            self.model.load_state_dict(cp_dict["generator"])
        else:
            self.model.load_state_dict(
                {
                    k.replace("generator.", ""): v
                    for k, v in cp_dict["state_dict"].items()
                    if k.startswith("generator.")
                }
            )

        self.model.eval()
        self.model.remove_weight_norm()

        self.mel_transform = PitchAdjustableMelSpectrogram(
            sample_rate=config["sampling_rate"],
            n_fft=config["n_fft"],
            win_length=config["win_length"],
            hop_length=config["hop_length"],
            f_min=config["f_min"],
            f_max=config["f_max"],
            n_mels=config["num_mels"],
        )

    @torch.no_grad()
    def spec2wav(self, mel, f0, key_shift=0):
        c = mel[None]
        f0 *= 2 ** (key_shift / 12)

        if self.use_natural_log is False:
            c = 2.30259 * c

        f0 = f0[None, None].to(c.dtype)
        y = self.model(c, f0).view(-1)

        return y

    @property
    def device(self):
        return next(self.model.parameters()).device

    def wav2spec(self, wav_torch, sr=None, key_shift=0, speed=1.0):
        if sr is None:
            sr = self.config["sampling_rate"]

        if sr != self.config["sampling_rate"]:
            _wav_torch = librosa.resample(
                wav_torch.cpu().numpy(),
                orig_sr=sr,
                target_sr=self.config["sampling_rate"],
            )
            wav_torch = torch.from_numpy(_wav_torch).to(wav_torch.device)

        mel_torch = self.mel_transform(wav_torch)[
            0
        ]  # , key_shift=key_shift, speed=speed)[0]
        mel_torch = dynamic_range_compression(mel_torch)

        if self.use_natural_log is False:
            mel_torch = 0.434294 * mel_torch

        return mel_torch
