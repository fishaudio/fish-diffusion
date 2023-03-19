from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from fish_diffusion.schedulers.warmup_cosine_scheduler import LambdaWarmUpCosineScheduler
from fish_diffusion.datasets.naive import NaiveSVCDataset\
from fish_diffusion.datasets.utils import (
    get_datasets_from_subfolder,
    get_speaker_map_from_subfolder,
)
from fish_diffusion.utils.pitch import pitch_to_log

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]
speaker_mapping = {}
val_mapping = {}
speaker_mapping = get_speaker_map_from_subfolder(
    "dataset/train", speaker_mapping
)  # Update speaker_mapping using subfolders in `dataset/train`.
val_mapping = get_speaker_map_from_subfolder("dataset/valid", val_mapping)
# This will update speaker_mapping to {'speaker0': 0, 'speaker': 1}
train_datasets = get_datasets_from_subfolder(
    "NaiveSVCDataset", "dataset/train", speaker_mapping
)  # Build datasets manually.
valid_datasets = get_datasets_from_subfolder(
    "NaiveSVCDataset", "dataset/valid", val_mapping
)  # Build datasets manually.

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",  # You want to contact multiple datasets
        datasets=train_datasets,
        # Are there any other ways to do this?
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",  # You want to contact multiple datasets
        datasets=valid_datasets,
        # Are there any other ways to do this?
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
        type="ParselMouthPitchExtractor",
    ),
)
augmentations = [
    dict(
        type="FixedPitchShifting",
        key_shifts=[-5.0, 5.0],
        probability=0.75,
    )
]
trainer = dict(
    devices=8,  # The number of GPUs that Every nodes has
    num_nodes=6,  # Total nodes
    max_steps=100000,  # It is recommended to make it lower when you use more GPUs
    val_check_interval=None,
    check_val_every_n_epoch=5,  # Steps val is suggested to disable,because the steps are only calcating in one rank,so Epoch val is better
)
# Learning rate setting override
lambda_func = LambdaWarmUpCosineScheduler(
    warm_up_steps=400,
    lr_min=8e-4,  # Too small value is not recommended if you have many GPUs to train
    lr_max=1.5e-3,  # Modify with your total batch_size,don't make it too high
    lr_start=5e-4,
    max_decay_steps=60000,
)
