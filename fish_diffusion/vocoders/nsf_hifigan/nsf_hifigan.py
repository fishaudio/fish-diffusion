import torch

import json
import os
from .models import Generator, AttrDict
from fish_diffusion.utils.audio import get_mel_from_audio
import pytorch_lightning as pl


class NsfHifiGAN(pl.LightningModule):
    def __init__(
        self,
        model_path: str = "checkpoints/nsf_hifigan/model",
        sample_rate: int = 44100,
        num_mels: int = 128,
        n_fft: int = 2048,
        win_size: int = 2048,
        hop_size: int = 512,
        fmin: int = 40,
        fmax: int = 16000,
    ):
        super().__init__()

        config_file = os.path.join(os.path.split(model_path)[0], "config.json")
        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.model = Generator(self.h)

        cp_dict = torch.load(model_path)
        self.model.load_state_dict(cp_dict["generator"])
        self.model.eval()
        self.model.remove_weight_norm()

        assert (
            self.h.sampling_rate == sample_rate
        ), f"Sampling rate mismatch: {self.h.sampling_rate} (vocoder)!= {sample_rate}"

        assert (
            self.h.num_mels == num_mels
        ), f"Number of mel bins mismatch: {self.h.num_mels} (vocoder) != {num_mels}"

        assert (
            self.h.n_fft == n_fft
        ), f"FFT size mismatch: {self.h.n_fft} (vocoder) != {n_fft}"

        assert (
            self.h.win_size == win_size
        ), f"Window size mismatch: {self.h.win_size} (vocoder) != {win_size}"

        assert (
            self.h.hop_size == hop_size
        ), f"Hop size mismatch: {self.h.hop_size} (vocoder) != {hop_size}"

        assert (
            self.h.fmin == fmin
        ), f"Minimum frequency mismatch: {self.h.fmin} (vocoder) != {fmin}"

        assert (
            self.h.fmax == fmax
        ), f"Maximum frequency mismatch: {self.h.fmax} (vocoder) != {fmax}"

    @torch.no_grad()
    def spec2wav(self, mel, f0):
        c = mel[None]
        f0 = f0[None].to(c.dtype)
        y = self.model(c, f0).view(-1)

        return y

    @property
    def device(self):
        return next(self.model.parameters()).device

    def wav2spec(self, wav_torch):
        mel_torch = get_mel_from_audio(
            audio=wav_torch,
            sample_rate=self.h.sampling_rate,
            n_fft=self.h.n_fft,
            win_length=self.h.win_size,
            hop_length=self.h.hop_size,
            f_min=self.h.fmin,
            f_max=self.h.fmax,
            n_mels=self.h.num_mels,
        )

        return mel_torch
