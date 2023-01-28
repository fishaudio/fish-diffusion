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
            residual_channels=384,
            use_linear_bias=True,
            dilation_cycle=4,
        ),
        keep_bins=1,  # It should be 128 if you are using the pretrained model
    ),
    vocoder=dict(
        use_natural_log=False,
    ),
)

preprocessing = dict(
    # You need to choose either "parselmouth" or "crepe" for pitch_extractor
    # But crepe seems buggy, I will debug it later
    pitch_extractor="parselmouth"
)
