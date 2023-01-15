from fish_diffusion.utils.pitch import pitch_to_scale

_base_ = [
    "./svc_hubert_soft.py",
]

model = dict(
    pitch_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=256,
        use_embedding=False,
        preprocessing=pitch_to_scale,
    ),
)
