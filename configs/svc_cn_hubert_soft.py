from fish_diffusion.datasets.audio_folder import AudioFolderDataset

_base_ = [
    "./svc_hubert_soft.py",
]

preprocessing = dict(
    text_features_extractor=dict(
        _delete_=True,
        type="ChineseHubertSoft",
        pretrained=True,
    ),
)
