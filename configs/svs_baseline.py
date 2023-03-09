# Warning: This config is developing, and subject to change.

from fish_diffusion.utils.dictionary import load_dictionary

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

speaker_mapping = {
    "default": 0,
}

dictionary, phonemes = load_dictionary("dictionaries/opencpop-extension.txt")

model = dict(
    type="DiffSinger",
    text_encoder=dict(
        _delete_=True,
        type="FastSpeech2Encoder",
        input_size=len(phonemes),
        hidden_size=256,
    ),
)

dataset = dict(
    _delete_=True,
    train=dict(
        type="NaiveSVSDataset",
        path="dataset/train",
        speaker_id=speaker_mapping["default"],
    ),
    valid=dict(
        type="NaiveSVSDataset",
        path="dataset/valid",
        speaker_id=speaker_mapping["default"],
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="OpenCpopTranscriptionToPhonemesDuration",
        phonemes=phonemes,
        transcription_path="dataset/transcriptions.txt",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
        f0_min=40.0,
        f0_max=2000.0,
    ),
    augmentations=[
        dict(
            type="RandomPitchShifting",
            key_shifts=[-5.0, 5.0],
            probability=1.5,
        ),
    ],
)
