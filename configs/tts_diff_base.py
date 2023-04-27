_base_ = [
    "./_base_/archs/diff_tts.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_tts.py",
]


preprocessing = dict(
    text_features_extractor=dict(
        type="PlainTextExtractor",
    ),
    pitch_extractor=dict(
        # ParselMouth is much faster than Crepe
        # However, Crepe may have better performance in some cases
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
    ),
)
