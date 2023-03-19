from fish_diffusion.utils.pitch import pitch_to_log

sampling_rate = 44100
hidden_size = 256

vocoder_config = {
    "resblock": "1",
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "upsample_rates": [8, 8, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
    "segment_size": 16384,
    "num_mels": 256,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "sampling_rate": 44100,
    "fmin": 40,
    "fmax": 16000,
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
        output_size=256,
        use_embedding=False,
    ),
    energy_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=256,
        use_embedding=False,
    ),
    encoder=vocoder_config,
)
