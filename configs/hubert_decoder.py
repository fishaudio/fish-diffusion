from fish_diffusion.datasets.naive import NaiveDenoiserDataset
from fish_diffusion.datasets.utils import (
    get_datasets_from_subfolder,
    get_speaker_map_from_subfolder,
)

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {"aria": 0}

speaker_mapping = get_speaker_map_from_subfolder("dataset/tts/genshin", speaker_mapping)
genshin_dataset = get_datasets_from_subfolder(
    "NaiveDenoiserDataset", "dataset/tts/genshin", speaker_mapping
)

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/aria",
                speaker_id=speaker_mapping["aria"],
            ),
        ]
        + genshin_dataset,
        collate_fn=NaiveDenoiserDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/valid/aria",
                speaker_id=speaker_mapping["aria"],
            ),
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/valid/aria",
                speaker_id=speaker_mapping["派蒙"],
            ),
        ],
        collate_fn=NaiveDenoiserDataset.collate_fn,
    ),
)

model = dict(
    text_encoder=dict(
        input_size=1024,
    ),
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
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
        type="QuantizedWhisper",
        # models=["TencentGameMate/chinese-hubert-large", "facebook/hubert-large-ll60k"],
    ),
    pitch_extractor=None,
    augmentations=[],
)
