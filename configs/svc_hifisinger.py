from pathlib import Path

from fish_diffusion.datasets.hifisinger import HiFiSVCDataset
from fish_diffusion.datasets.utils import get_datasets_from_subfolder
from fish_diffusion.utils.pitch import pitch_to_log

_base_ = [
    "./_base_/archs/hifi_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {
    "aria": 0,
    "opencpop": 1,
    "lengyue": 2,
}


mixin_datasets = []

mixin_dataset = Path("dataset/svc-mixin-dataset/M4Singer")
for sub_path in mixin_dataset.iterdir():
    splitter = "_" if "_" in sub_path.name else "#"
    singer_id = sub_path.name.split(splitter)[0]
    speaker_name = f"{mixin_dataset.name}-{singer_id}"

    if speaker_name not in speaker_mapping:
        speaker_mapping[speaker_name] = len(speaker_mapping)

    mixin_datasets.append(
        dict(
            type="HiFiSVCDataset",
            path=str(sub_path),
            speaker_id=speaker_mapping[speaker_name],
            segment_size=16384,
        )
    )

# Process SVC mixin datasets
dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "HiFiSVCDataset", "dataset/train", speaker_mapping, segment_size=16384
        )
        + mixin_datasets,
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default valid dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "HiFiSVCDataset", "dataset/valid", speaker_mapping
        ),
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
)


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
            probability=0.75,
        ),
    ],
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None
)
