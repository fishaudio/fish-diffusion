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
        _delete_=True,
    ),
    pitch_encoder=dict(
        _delete_=True,
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
