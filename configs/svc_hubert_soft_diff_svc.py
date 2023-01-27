from fish_diffusion.utils.pitch import pitch_to_coarse

_base_ = [
    "./svc_hubert_soft.py",
]

hidden_size = 256

model = dict(
    type="DiffSVC",
    speaker_encoder=dict(
        # This is currently not used, all params will be zeroed
        type="NaiveProjectionEncoder",
        input_size=10,
        output_size=hidden_size,
        use_embedding=True,
    ),
    pitch_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=300,
        output_size=hidden_size,
        use_embedding=True,
        preprocessing=pitch_to_coarse,
    ),
    text_encoder=dict(
        _delete_=True,
        type="IdentityEncoder",
    ),
    diffusion=dict(
        denoiser=dict(
            use_linear_bias=True,
            residual_channels=384,
        ),
        keep_bins=1,  # It should be 128 if you are using the pretrained model
    ),
)
