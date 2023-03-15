import json
from pathlib import Path
from typing import Optional

import librosa
import pytorch_lightning as pl
import torch

from fish_diffusion.utils.audio import dynamic_range_compression
from fish_diffusion.utils.pitch_adjustable_mel import PitchAdjustableMelSpectrogram

from ..builder import VOCODERS
from ..nsf_hifigan.models import AttrDict
from .models import Generator


@VOCODERS.register_module()
class ISTFTNet(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/istft_net/g_00045000",
        config_file: Optional[str] = None,
        use_natural_log: bool = True,
        **kwargs,
    ):
        super().__init__()

        if config_file is None:
            config_file = Path(checkpoint_path).parent / "config.json"

        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.model = Generator(self.h)
        self.use_natural_log = use_natural_log

        cp_dict = torch.load(checkpoint_path)

        if "state_dict" not in cp_dict:
            self.model.load_state_dict(cp_dict["generator"], map_location="cpu")
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
            sample_rate=self.h.sampling_rate,
            n_fft=self.h.n_fft,
            win_length=self.h.win_size,
            hop_length=self.h.hop_size,
            f_min=self.h.fmin,
            f_max=self.h.fmax,
            n_mels=self.h.num_mels,
        )

        # Validate kwargs if any
        if "mel_channels" in kwargs:
            kwargs["num_mels"] = kwargs.pop("mel_channels")

        for k, v in kwargs.items():
            if getattr(self.h, k, None) != v:
                raise ValueError(f"Incorrect value for {k}: {v}")

    @torch.no_grad()
    def spec2wav(self, mel, f0):
        c = mel[None]

        if self.use_natural_log is False:
            c = 2.30259 * c

        f0 = f0[None].to(c.dtype)
        spec, phase = self.model(c)

        y = torch.istft(
            spec * torch.exp(phase * 1j),
            n_fft=self.h.gen_istft_n_fft,
            hop_length=self.h.gen_istft_hop_size,
            win_length=self.h.gen_istft_n_fft,
            window=self.hanning_window,
        )

        return y[0]

    @torch.no_grad()
    def wav2spec(self, wav_torch, sr=None, key_shift=0, speed=1.0):
        if sr is None:
            sr = self.h.sampling_rate

        if sr != self.h.sampling_rate:
            _wav_torch = librosa.resample(
                wav_torch.cpu().numpy(), orig_sr=sr, target_sr=self.h.sampling_rate
            )
            wav_torch = torch.from_numpy(_wav_torch).to(wav_torch.device)

        mel_torch = self.mel_transform(wav_torch, key_shift=key_shift, speed=speed)[0]
        mel_torch = dynamic_range_compression(mel_torch)

        if self.use_natural_log is False:
            mel_torch = 0.434294 * mel_torch

        return mel_torch
