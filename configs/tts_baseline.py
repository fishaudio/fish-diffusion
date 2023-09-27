# Warning: This config is developing, and subject to change.

from pathlib import Path

from fish_diffusion.datasets.naive import NaiveTTSDataset
from fish_diffusion.schedulers.warmup_cosine_scheduler import (
    LambdaWarmUpCosineScheduler,
)

_base_ = [
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

speakers = []

# Process SVC mixin datasets
mixin_datasets = [
    ("LibriTTS-100", "dataset/LibriTTS/train-clean-100"),
    ("LibriTTS-360", "dataset/LibriTTS/train-clean-360"),
    ("LibriTTS-500", "dataset/LibriTTS/train-other-500"),
]
train_datasets = []

for name, path in mixin_datasets:
    for speaker_path in sorted(Path(path).iterdir()):
        if not any(speaker_path.rglob("*.npy")):
            continue

        speaker_name = f"{name}-{speaker_path.name}"
        if speaker_name not in speakers:
            speakers.append(speaker_name)

        train_datasets.append(
            dict(
                type="NaiveTTSDataset",
                path=str(speaker_path),
                speaker_id=speaker_name,
                cache_list=True,
            )
        )

# Sort speakers
speakers.sort()
speaker_mapping = {speaker: i for i, speaker in enumerate(speakers)}

for dataset in train_datasets:
    dataset["speaker_id"] = speaker_mapping[dataset["speaker_id"]]

# Config model
sampling_rate = 44100
mel_channels = 128
bert_dim = 768
gradient_checkpointing = False

model = dict(
    type="GradTTS",
    gradient_checkpointing=gradient_checkpointing,
    diffusion=dict(
        type="GaussianDiffusion",
        mel_channels=mel_channels,
        noise_schedule="linear",
        timesteps=1000,
        max_beta=0.01,
        s=0.008,
        noise_loss="smoothed-l1",
        denoiser=dict(
            type="WaveNetDenoiser",
            mel_channels=mel_channels,
            d_encoder=bert_dim,
            residual_channels=512,
            residual_layers=20,
            dilation_cycle=4,
            use_linear_bias=True,
        ),
        sampler_interval=10,
        spec_min=[-5],
        spec_max=[0],
    ),
    speaker_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=10000,  # len(speaker_mapping),
        output_size=bert_dim,
        use_embedding=True,
    ),
    text_encoder=dict(
        type="BertEncoder",
        model_name="bert-base-cased",
        pretrained=True,
    ),
    duration_predictor=dict(
        type="NaiveProjectionEncoder",
        input_size=bert_dim,
        output_size=1,
    ),
    vocoder=dict(
        type="ADaMoSHiFiGANV1",
        use_natural_log=False,
        checkpoint_path="checkpoints/adamos/convnext_hifigan_more_supervised_001560000.ckpt",
    ),
)

dataset = dict(
    _delete_=True,
    train=dict(
        type="ConcatDataset",
        datasets=train_datasets,
        collate_fn=NaiveTTSDataset.collate_fn,
    ),
    valid=dict(
        type="SampleDataset",
        num_samples=8,
        dataset=dict(
            type="ConcatDataset",
            datasets=train_datasets,
            collate_fn=NaiveTTSDataset.collate_fn,
        ),
        collate_fn=NaiveTTSDataset.collate_fn,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=16,
    ),
    valid=dict(
        batch_size=8,
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="BertTokenizer",
        model_name="bert-base-cased",
        label_suffix=".normalized.txt",
    ),
)

lambda_func = LambdaWarmUpCosineScheduler(
    warm_up_steps=10000,
    val_final=1e-5,
    val_base=4e-5,
    val_start=0,
    max_decay_steps=1000000,
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1.0,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-6,
)

scheduler = dict(
    _delete_=True,
    type="LambdaLR",
    lr_lambda=lambda_func,
)
