from fish_diffusion.datasets.hifisinger import HiFiSVCDataset
from fish_diffusion.datasets.utils import (
    get_speaker_map_from_subfolder,
    get_datasets_from_subfolder,
)

_base_ = [
    "./svc_hifisinger_finetune.py",
]

speaker_mapping = {}

speaker_mapping = get_speaker_map_from_subfolder(
    "dataset/train", speaker_mapping
)  # Update speaker_mapping using subfolders in `dataset/train`.

train_datasets = get_datasets_from_subfolder(
    "HiFiSVCDataset", "dataset/train", speaker_mapping, segment_size=16384
)  # Build datasets manually.
valid_datasets = get_datasets_from_subfolder(
    "HiFiSVCDataset", "dataset/valid", speaker_mapping, segment_size=-1
)  # Build datasets manually.

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",  # You want to contact multiple datasets
        datasets=train_datasets,
        # Are there any other ways to do this?
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",  # You want to contact multiple datasets
        datasets=valid_datasets,
        # Are there any other ways to do this?
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
)

model = dict(
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
)
