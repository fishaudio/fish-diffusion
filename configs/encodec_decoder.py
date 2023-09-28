_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {
    "default": 0,
}

dataset = dict(
    train=dict(
        type="NaiveDenoiserDataset",
        path="dataset/tts",
        speaker_id=0,
    ),
    valid=dict(
        type="NaiveDenoiserDataset",
        path="dataset/tts/valid",
        speaker_id=0,
    ),
)

model = dict(
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=128,
        output_size=256,
    ),
    speaker_encoder=dict(
        _delete_=True,
    ),
    pitch_encoder=dict(
        _delete_=True,
    ),
    vocoder=dict(
        _delete_=True,
        type="ADaMoSHiFiGANV1",
        use_natural_log=False,
        checkpoint_path="checkpoints/adamos/convnext_hifigan_more_supervised_001560000.ckpt",
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="Encodec",
        model="facebook/encodec_24khz",
        first_codebook_only=True,
    ),
    pitch_extractor=None,
    augmentations=[],
)
