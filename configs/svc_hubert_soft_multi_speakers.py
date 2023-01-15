from fish_diffusion.datasets.audio_folder import AudioFolderDataset

_base_ = [
    "./svc_hubert_soft.py",
]

dataset = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="AudioFolderDataset",
                path="dataset/speaker_0",
                speaker_id=0,
            ),
            dict(
                type="AudioFolderDataset",
                path="dataset/speaker_1",
                speaker_id=1,
            ),
        ],
        # Are there any other ways to do this?
        collate_fn=AudioFolderDataset.collate_fn,
    ),
    valid=dict(
        type="AudioFolderDataset",
        path="dataset/valid",
        speaker_id=0,
    ),
)

model = dict(
    speaker_encoder=dict(
        input_size=2,  # 2 speakers
    ),
)
