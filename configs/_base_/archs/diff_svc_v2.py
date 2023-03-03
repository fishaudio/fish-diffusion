"""
DiffSVC architecture with WaveNet denoiser and NSF-HiFiGAN vocoder.

Comparing to v1, this version 
- Doesn't need spec stats anymore.
- Added dilation cycle to WaveNet denoiser.
- Used the log10 mel spectrogram.
- Better matching DiffSinger architecture.
"""

from fish_diffusion.utils.pitch import pitch_to_scale

sampling_rate = 44100
mel_channels = 128
hidden_size = 256

model = dict(
    type="DiffSVC",
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
            dilation_cycle=4,
            use_linear_bias=True,
        ),
        sampler_interval=10,
        spec_min=[-5],
        spec_max=[0],
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
    pitch_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=hidden_size,
        use_embedding=False,
        preprocessing=pitch_to_scale,
    ),
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
        sampling_rate=sampling_rate,
        mel_channels=mel_channels,
        use_natural_log=False,
    ),
)
