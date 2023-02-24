from functools import partial

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from fish_diffusion.utils.pitch import pitch_to_scale

sampling_rate = 44100

trainer = dict(
    accelerator="gpu",
    devices=-1,
    max_epochs=-1,
    precision=16,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_epochs=5,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
    strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="nccl"),
)

model = dict(
    type="NSF-HiFiGAN",
    config="tools/nsf_hifigan/config_v1.json",
    # The following code are used for preprocessing
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
    ),
)

dataset = dict(
    train=dict(
        type="VOCODERDataset",
        path="/mnt/nvme0/diff-wave-data/train",
        segment_size=16384,
    ),
    valid=dict(
        type="VOCODERDataset",
        path="/mnt/nvme0/diff-wave-data/valid",
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
    text_features_extractor=None,
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
    ),
)
