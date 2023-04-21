from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from fish_diffusion.datasets.naive import NaiveSVCDataset
from fish_diffusion.utils.pitch import pitch_to_log

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine_finetune.py",
    "./_base_/datasets/naive_svc.py",
]

speaker_mapping = {
    "Placeholder": 0,
}

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="NaiveSVCDataset",
                path="dataset/train",
                speaker_id=speaker_mapping["Placeholder"],
            ),
        ],
        # Are there any other ways to do this?
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default valid dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="NaiveSVCDataset",
                path="dataset/valid",
                speaker_id=speaker_mapping["Placeholder"],
            ),
        ],
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
)

model = dict(
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=256,
        output_size=256,
    ),
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
    pitch_encoder=dict(
        preprocessing=pitch_to_log,
    ),
    pitch_shift_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=256,
        use_embedding=False,
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ContentVec",
    ),
    pitch_extractor=dict(
        type="CrepePitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
    ),
    energy_extractor=dict(
        type="RMSEnergyExtractor",
    ),
    augmentations=[
        dict(
            type="RandomPitchShifting",
            key_shifts=[-5.0, 5.0],
            probability=1.5,
        ),
    ],
)

# The following trainer val and save checkpoints every 1000 steps
trainer = dict(
    val_check_interval=1000,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=1000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
