from fish_diffusion.utils.pitch import pitch_to_log

sampling_rate = 44100
hidden_size = 256

vocoder_config = {
    "type": "HiFiGAN",
    "sampling_rate": sampling_rate,
    # Model config
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
    # The following parameters are used for validation
    "num_mels": 256,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "fmin": 40,
    "fmax": 16000,
    # The following parameters are used for training
    "multi_scale_mels": [
        (2048, 512, 2048),  # (n_fft, hop_size, win_size)
        (2048, 270, 1080),
        (4096, 540, 2160),
    ],
    "multi_scale_stfts": [
        (512, 50, 240),  # (n_fft, hop_size, win_size)
        (1024, 120, 600),
        (2048, 240, 1200),
    ],
}


model = dict(
    type="HiFiSVC",
    hidden_size=hidden_size,
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
        preprocessing=pitch_to_log,
    ),
    pitch_shift_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=hidden_size,
        use_embedding=False,
    ),
    energy_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=hidden_size,
        use_embedding=False,
    ),
    encoder=vocoder_config,
)
