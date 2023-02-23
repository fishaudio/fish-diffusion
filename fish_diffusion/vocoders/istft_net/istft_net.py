import json
import os

import librosa
import pytorch_lightning as pl
import torch

from fish_diffusion.utils.audio import get_mel_from_audio

from ..builder import VOCODERS
from ..nsf_hifigan import NsfHifiGAN
from .mel import mel_spectrogram
from .models import AttrDict, Generator


@VOCODERS.register_module()
class ISTFTNet(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/istft_net/g_00045000",
        use_natural_log: bool = True,
    ):
        super().__init__()

        config_file = os.path.join(os.path.split(checkpoint_path)[0], "config.json")
        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.model = Generator(self.h)
        self.use_natural_log = use_natural_log

        cp_dict = torch.load(checkpoint_path)

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
        self.register_buffer(
            "hanning_window", torch.hann_window(self.h.gen_istft_n_fft)
        )

    @torch.no_grad()
    def spec2wav(self, mel, f0):
        c = mel[None]

        if self.use_natural_log is False:
            c = 2.30259 * c

        f0 = f0[None].to(c.dtype)
        spec, phase = self.model(c)

        audio_1 = spec * torch.exp(phase * 1j)
        audio_1 = torch.view_as_real(audio_1)
        print(audio_1.shape, "Audio 0", torch.max(audio_1), torch.min(audio_1))
        y = torch.istft(
            spec * torch.exp(phase * 1j),
            n_fft=self.h.gen_istft_n_fft,
            hop_length=self.h.gen_istft_hop_size,
            win_length=self.h.gen_istft_n_fft,
            window=self.hanning_window,
        )

        # * 32768.0
        print(y.shape)

        return y[0]

    @torch.no_grad()
    def wav2spec(self, audio, sr=None):
        if sr is None:
            sr = self.h.sampling_rate

        if sr != self.h.sampling_rate:
            _wav_torch = librosa.resample(
                audio.cpu().numpy(), orig_sr=sr, target_sr=self.h.sampling_rate
            )
            audio = torch.from_numpy(_wav_torch).to(audio.device)

        # audio = audio / 32768.0

        # mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
        #                       self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
        #                       center=False)

        audio_1 = torch.stft(
            audio,
            n_fft=self.h.gen_istft_n_fft,
            hop_length=self.h.gen_istft_hop_size,
            win_length=self.h.gen_istft_n_fft,
            window=self.hanning_window,
            return_complex=True,
        )
        audio_1 = torch.view_as_real(audio_1)
        print(audio_1.shape, "Audio 1", torch.max(audio_1), torch.min(audio_1))

        x = mel_spectrogram(
            audio,
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax,
            center=False,
        )[0]

        if self.use_natural_log is False:
            x = 0.434294 * x

        return x
