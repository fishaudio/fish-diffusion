from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

_base_ = [
    "./_base_/schedulers/exponential.py",
]

sampling_rate = 44100
num_mels = 128
n_fft = 2048
hop_length = 256
win_length = 2048

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
    type="RefineGAN",
    f_min=40.0,
    f_max=16000.0,
    generator=dict(
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        downsample_rates=[2, 2, 8, 8],
        upsample_rates=[8, 8, 2, 2],
        leaky_relu_slope=0.2,
        num_mels=num_mels,
        start_channels=16,
    ),
    # Discriminators
    mpd=dict(periods=[2, 3, 5, 7, 11]),
    mrd=dict(
        resolutions=[
            [1024, 120, 600],
            [2048, 240, 1200],
            [512, 50, 240],
        ],
    ),
    multi_scale_mels=[
        (n_fft, hop_length, win_length),
        (2048, 270, 1080),
        (4096, 540, 2160),
    ],
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
