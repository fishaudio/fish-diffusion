from functools import partial

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from fish_diffusion.utils.pitch import pitch_to_scale

sampling_rate = 44100
hop_length = 256

trainer = dict(
    accelerator="gpu",
    devices=-1,
    max_epochs=-1,
    precision="32",
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_on_train_epoch_end=False,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
    strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="nccl"),
)

model = dict(
    type="NSF-HiFiGAN",
    config="tools/nsf_hifigan/config_v1_256.json",
    # The following code are used for preprocessing
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
    ),
)

dataset = dict(
    train=dict(
        type="NaiveVOCODERDataset",
        path="/fs/nexus-scratch/lengyue/vocoder-data/train",
        segment_size=32768,
        pitch_shift=[-12, 12],
        loudness_shift=[0.1, 0.9],
        hop_length=hop_length,
        sampling_rate=sampling_rate,
    ),
    valid=dict(
        type="NaiveVOCODERDataset",
        path="/fs/nexus-scratch/lengyue/vocoder-data/valid",
        segment_size=None,
        pitch_shift=None,
        loudness_shift=None,
        hop_length=hop_length,
        sampling_rate=sampling_rate,
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
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)

preprocessing = dict(
    pitch_extractor=dict(
        type="HarvestPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
        hop_length=hop_length,
    ),
)
