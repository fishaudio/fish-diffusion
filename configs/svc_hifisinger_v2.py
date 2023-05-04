from pathlib import Path

from fish_diffusion.datasets.hifisinger import HiFiSVCDataset
from fish_diffusion.datasets.utils import get_datasets_from_subfolder

_base_ = [
    "./_base_/archs/hifi_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
    "./_base_/datasets/hifi_svc.py",
]

speaker_mapping = {
    "aria": 0,
    "opencpop": 1,
    "lengyue": 2,
}

mixin_datasets = []

# For pretraining
base_path = Path("dataset/svc-mixin-dataset")
for mixin_dataset in sorted(base_path.iterdir()):
    for sub_path in mixin_dataset.iterdir():
        splitter = "_" if "_" in sub_path.name and "#" not in sub_path.name else "#"
        singer_id = sub_path.name.split(splitter)[0]
        speaker_name = f"{mixin_dataset.name}-{singer_id}"

        if speaker_name not in speaker_mapping:
            speaker_mapping[speaker_name] = len(speaker_mapping)

        mixin_datasets.append(
            dict(
                type="HiFiSVCDataset",
                path=str(sub_path),
                speaker_id=speaker_mapping[speaker_name],
                segment_size=32768,
                hop_length=256,
            )
        )

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "HiFiSVCDataset",
            "dataset/train",
            speaker_mapping,
            segment_size=32768,
            hop_length=256,
        ),
        # + mixin_datasets,
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=get_datasets_from_subfolder(
            "HiFiSVCDataset",
            "dataset/valid",
            speaker_mapping,
            segment_size=-1,
            hop_length=256,
        ),
        collate_fn=HiFiSVCDataset.collate_fn,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=10,
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
        output_layer=None,
        use_projection=False,
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=1600.0,
        hop_length=256,
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
    gradient_clip_val=None,
    max_steps=1000000,
    precision="32-true",
)
