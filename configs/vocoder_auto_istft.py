from functools import partial

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from fish_diffusion.utils.pitch import pitch_to_scale

_base_ = [
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
]

sampling_rate = 44100
num_mels = 128
n_fft = 2048
hop_length = 512
win_length = 2048

model = dict(
    type="AutoVocoder",
    encoder=dict(
        n_blocks=11,
        latent_dim=256,
        latent_dropout=0.1,
        filter_length=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ),
    decoder=dict(
        n_blocks=11,
        latent_dim=256,
        filter_length=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    ),
    discriminator_periods=[3, 5, 7, 11, 17, 23, 37],
    # Params used for training
    multi_scale_mels=[
        (2048, 512, 2048),  # (n_fft, hop_size, win_size)
        (2048, 270, 1080),
        (4096, 540, 2160),
    ],
    multi_scale_stfts=[
        (512, 50, 240),  # (n_fft, hop_size, win_size)
        (1024, 120, 600),
        (2048, 240, 1200),
    ],
    # The following code are used for preprocessing
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
    ),
)

dataset = dict(
    train=dict(
        type="NaiveAudioDataset",
        path="/mnt/nvme1/vocoder-dataset/train",
        segment_size=16384,
    ),
    valid=dict(
        type="NaiveAudioDataset",
        path="/mnt/nvme1/vocoder-dataset/valid",
        segment_size=-1,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=20,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=20,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    ),
)

preprocessing = dict(
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
    ),
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None,
    max_steps=1000000,
    precision=32,  # Many iSTFTs are used in the model, so 16-bit precision is not enough
)
