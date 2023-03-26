from fish_diffusion.datasets.hifisinger import HiFiSVCDataset
from fish_diffusion.datasets.utils import get_datasets_from_subfolder

_base_ = [
    "./_base_/archs/hifi_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/exponential.py",
    "./_base_/datasets/hifi_svc.py",
]

speaker_mapping = {
    "Placeholder": 0,
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
            probability=0.75,
        ),
    ],
)

trainer = dict(
    # Disable gradient clipping, which is not supported by custom optimization
    gradient_clip_val=None,
    max_steps=1000000,
)
