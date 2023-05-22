from fish_diffusion.utils.pitch import pitch_to_log

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {k: str(k) for k in range(489)}

model = dict(
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1024,
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
    vocoder=dict(
        checkpoint_path="checkpoints/nsf_hifigan/model",
        config_file="checkpoints/nsf_hifigan/config.json",
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ChineseHubert", model="TencentGameMate/chinese-hubert-large"
    ),
    pitch_extractor=dict(
        type="CrepePitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
    ),
    augmentations=[],
)
