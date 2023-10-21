from fish_diffusion.datasets.naive import NaiveDenoiserDataset

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {"default": 0, "aria": 1}

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
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/vctk",
                speaker_id=0,
            ),
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/genshin",
                speaker_id=0,
            ),
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/libritts",
                speaker_id=0,
            ),
            dict(
                type="NaiveDenoiserDataset",
                path="dataset/tts/wenet-speech-vocals",
                speaker_id=0,
            ),
        ],
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
                path="dataset/tts/valid/vctk",
                speaker_id=0,
            ),
        ],
        collate_fn=NaiveDenoiserDataset.collate_fn,
    ),
)

model = dict(
    text_encoder=dict(
        type="MLPVectorQuantizeEncoder",
        input_size=1024 * 2,  # Dual hubert
        output_size=256,
        dim=512,
        codebook_size=4096,
        threshold_ema_dead_code=2,
        use_cosine_sim=True,
    ),
    speaker_encoder=dict(
        input_size=10,
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
        type="EnsembleHubert",
        models=["TencentGameMate/chinese-hubert-large", "facebook/hubert-large-ll60k"],
    ),
    pitch_extractor=None,
    augmentations=[],
)
