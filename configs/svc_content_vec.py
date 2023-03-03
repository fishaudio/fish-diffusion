_base_ = [
    "./svc_hubert_soft.py",
]

preprocessing = dict(
    text_features_extractor=dict(
        _delete_=True,
        type="ContentVec",
    ),
)
