
sampling_rate = 16000
mel_channels = 128
hidden_size = 256

model = dict(
    type="DiffTTS",
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
            d_encoder=hidden_size,
            residual_channels=512,
            residual_layers=20,
        ),
        spec_stats_path="dataset/stats.json",
        sampler_interval=10,
    ),
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=256,
        output_size=hidden_size,
    ),
    speaker_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=10,
        output_size=hidden_size,
        use_embedding=True,
    ),
)