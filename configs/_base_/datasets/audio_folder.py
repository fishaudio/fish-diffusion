dataset = dict(
    train=dict(
        type="AudioFolderDataset",
        path="dataset/aria",
        speaker_mapping="dataset/speakers.json",
    ),
    valid=dict(
        type="AudioFolderDataset",
        path="dataset/valid",
        speaker_mapping="dataset/speakers.json",
    ),
)

dataloader = dict(
    train=dict(
        batch_size=20,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=20,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)
