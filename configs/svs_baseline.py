# Warning: This config is developing, and subject to change.

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

phonemes = [
    "AP",
    "SP",
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
]

preprocessing = dict(
    text_features_extractor=dict(
        type="OpenCpopTranscriptionToPhonemesDuration",
        phonemes=phonemes,
        transcription_path="dataset/transcriptions.txt",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
    ),
)

model = dict(
    type="DiffSinger",
    text_encoder=dict(
        _delete_=True,
        type="NaiveProjectionEncoder",
        input_size=len(phonemes) * 2 + 2,
        output_size=256,
    ),
    diffusion=dict(
        max_beta=0.02,
    ),
)

dataset = dict(
    _delete_=True,
    train=dict(
        type="AudioFolderDataset",
        path="dataset/diff-singer/train",
        speaker_id=0,
    ),
    valid=dict(
        type="AudioFolderDataset",
        path="dataset/diff-singer/valid",
        speaker_id=0,
    ),
)
