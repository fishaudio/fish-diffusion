from functools import partial

import numpy as np

from fish_diffusion.utils.pitch import pitch_to_coarse

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/step.py",
    "./_base_/datasets/naive_svc.py",
]

hidden_size = 256

model = dict(
    type="DiffSVC",
    speaker_encoder=dict(
        _delete_=True,
        # This is currently not used, all params will be zeroed
        type="NaiveProjectionEncoder",
        input_size=10,
        output_size=hidden_size,
        use_embedding=True,
    ),
    pitch_encoder=dict(
        _delete_=True,
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
        ),
        spec_min=[-5] * 128,
        spec_max=[0] * 128,
    ),
)

preprocessing = dict(
    # You need to choose either "parselmouth" or "crepe" for pitch_extractor
    pitch_extractor=dict(
        type="CrepePitchExtractor",
        f0_min=40.0,
        f0_max=1100.0,
        keep_zeros=False,
    ),
    text_features_extractor=dict(
        type="HubertSoft",
    ),
)
