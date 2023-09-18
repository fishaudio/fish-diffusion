# Warning: This config is developing, and subject to change.

_base_ = [
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

speaker_mapping = {
    "default": 0,
}

sampling_rate = 44100
mel_channels = 128
bert_dim = 1024

model = dict(
    type="GradTTS",
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
            d_encoder=mel_channels,
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
        input_size=10,
        output_size=bert_dim,
        use_embedding=True,
    ),
    text_encoder=dict(
        type="BertEncoder",
        model_name="xlm-roberta-large",
        pretrained=True,
    ),
    mel_encoder=dict(
        type="TransformerEncoder",
        input_size=bert_dim,
        output_size=mel_channels,
        hidden_size=bert_dim,
        num_layers=4,
    ),
    duration_predictor=dict(
        type="TransformerEncoder",
        input_size=bert_dim,
        output_size=1,
        hidden_size=bert_dim,
        num_layers=1,
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
        type="NaiveTTSDataset",
        path="dataset/tts/train",
        speaker_id=speaker_mapping["default"],
    ),
    valid=dict(
        type="NaiveTTSDataset",
        path="dataset/tts/valid",
        speaker_id=speaker_mapping["default"],
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="BertTokenizer",
        model_name="xlm-roberta-large",
        transcription_path="dataset/tts/aria.list",
    ),
)
