import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from fish_diffusion.schedulers.warmup_cosine_scheduler import (
    LambdaWarmUpCosineScheduler,
)
from fish_diffusion.utils.pitch import pitch_to_scale
import sys

from fish_diffusion.modules.vocoders import NsfHifiGAN

# def pitch_to_scale(pitch):
#     return pitch_to_scale(pitch)


def create_ddp_strategy():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return {
            "find_unused_parameters": True,
            "process_group_backend": "nccl" if sys.platform != "win32" else "gloo",
            "gradient_as_bucket_view": True,
            "ddp_comm_hook": default.fp16_compress_hook,
        }
    else:
        return None


def create_lr_lambda():
    return LambdaWarmUpCosineScheduler(
        warm_up_steps=1000,
        lr_min=1e-4,
        lr_max=8e-4,
        lr_start=1e-5,
        max_decay_steps=150000,
    )


def register_resolvers(OmegaConf):
    OmegaConf.register_new_resolver("mel_channels", lambda: 128)
    OmegaConf.register_new_resolver("sampling_rate", lambda: 44100)
    OmegaConf.register_new_resolver("hidden_size", lambda: 256)
    OmegaConf.register_new_resolver("n_fft", lambda: 2048)
    OmegaConf.register_new_resolver("hop_length", lambda: 256)
    OmegaConf.register_new_resolver("win_length", lambda: 2048)
    OmegaConf.register_new_resolver("pitch_to_scale", lambda: pitch_to_scale)
    OmegaConf.register_new_resolver("create_ddp_strategy", create_ddp_strategy)
    OmegaConf.register_new_resolver("create_lr_lambda", create_lr_lambda)
    OmegaConf.register_new_resolver("NsfHifiGAN", NsfHifiGAN)
