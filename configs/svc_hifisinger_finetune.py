from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

_base_ = [
    "./_base_/archs/hifi_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
    "./_base_/datasets/hifi_svc.py",
]

speaker_mapping = {
    "placeholder": 0,
}

model = dict(
    type="HiFiSVC",
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ContentVec",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
    ),
    energy_extractor=dict(
        type="RMSEnergyExtractor",
    ),
    augmentations=[
        dict(
            type="FixedPitchShifting",
            key_shifts=[-5.0, 5.0],
            probability=1.5,
        ),
    ],
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=1000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
