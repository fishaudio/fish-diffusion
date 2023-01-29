from functools import partial

import numpy as np

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
        # Since the pretrained model uses a 40.0 Hz minimum pitch,
        preprocessing=partial(
            pitch_to_coarse, f0_mel_min=1127 * np.log(1 + 40.0 / 700)
        ),
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
    pitch_extractor=dict(
        _delete_=True,
        type="ParselMouthPitchExtractor",
        f0_min=40.0,
        f0_max=1100.0,
    )
)
