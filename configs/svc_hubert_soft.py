from fish_diffusion.utils.pitch import pitch_to_scale

_base_ = [
    "./_base_/archs/diff_svc.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/step.py",
    "./_base_/datasets/audio_folder.py",
]


preprocessing = dict(
    text_features_extractor=dict(
        type="HubertSoft",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
    ),
)
