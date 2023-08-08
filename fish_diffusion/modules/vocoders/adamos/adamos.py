import librosa
import pytorch_lightning as pl
import torch
from loguru import logger

from fish_diffusion.utils.audio import dynamic_range_compression
from fish_diffusion.utils.pitch_adjustable_mel import PitchAdjustableMelSpectrogram

from ..builder import VOCODERS
from .encoder import ConvNeXtEncoder
from .hifigan import HiFiGANGenerator


@VOCODERS.register_module()
class ADaMoSHiFiGANV1(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/adamos/convnext_hifigan_more_supervised_001280000.ckpt",
        use_natural_log: bool = True,
    ):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=128,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0,
            kernel_sizes=(7,),
        )

        self.head = HiFiGANGenerator(
            hop_length=512,
            upsample_rates=(4, 4, 2, 2, 2, 2, 2),
            upsample_kernel_sizes=(8, 8, 4, 4, 4, 4, 4),
            resblock_kernel_sizes=(3, 7, 11, 13),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)),
            num_mels=512,
            upsample_initial_channel=1024,
            use_template=False,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        )
        self.use_natural_log = use_natural_log
        self.sampling_rate = 44100

        ckpt_state = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        if any(k.startswith("generator.") for k in ckpt_state):
            ckpt_state = {
                k.replace("generator.", ""): v
                for k, v in ckpt_state.items()
                if k.startswith("generator.")
            }

        self.load_state_dict(ckpt_state)
        self.eval()
        self.head.remove_weight_norm()

        self.mel_transform = PitchAdjustableMelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=40,
            f_max=16000,
            n_mels=128,
        )

    @torch.no_grad()
    def spec2wav(self, mel, f0=None, key_shift=0):
        c = mel[None]

        if key_shift != 0:
            logger.info(
                "key_shift is not for ADaMoSHiFiGANV1, since it is not pitch-conditioned"
            )

        if self.use_natural_log is False:
            c = 2.30259 * c

        y = self.backbone(c)
        y = self.head(y).view(-1)

        return y

    @property
    def device(self):
        return next(self.model.parameters()).device

    def wav2spec(self, wav_torch, sr=None, key_shift=0, speed=1.0):
        if sr is None:
            sr = self.sampling_rate

        if sr != self.sampling_rate:
            _wav_torch = librosa.resample(
                wav_torch.cpu().numpy(), orig_sr=sr, target_sr=self.sampling_rate
            )
            wav_torch = torch.from_numpy(_wav_torch).to(wav_torch.device)

        mel_torch = self.mel_transform(wav_torch, key_shift=key_shift, speed=speed)[0]
        mel_torch = dynamic_range_compression(mel_torch)

        if self.use_natural_log is False:
            mel_torch = 0.434294 * mel_torch

        return mel_torch
