from fish_diffusion.utils.pitch import pitch_to_log

sampling_rate = 44100
hidden_size = 256
num_mels = 128
n_fft = 2048
hop_length = 256
win_length = 2048

model = dict(
    type="HiFiSVC",
    hidden_size=hidden_size,
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=768,
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
    encoder=dict(
        type="RefineGAN",
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        downsample_rates=[2, 2, 8, 8],
        upsample_rates=[8, 8, 2, 2],
        leaky_relu_slope=0.2,
        num_mels=hidden_size,
        start_channels=16,
    ),
    # Discriminators
    mpd=dict(periods=[2, 3, 5, 7, 11]),
    mrd=dict(
        resolutions=[
            [1024, 120, 600],
            [2048, 240, 1200],
            [512, 50, 240],
        ],
    ),
    multi_scale_mels=[
        (n_fft, hop_length, win_length),
        (2048, 270, 1080),
        (4096, 540, 2160),
    ],
)
