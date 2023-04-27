dataset = dict(
    train=dict(
        type="NaiveTTSDataset",
        path="dataset/train",
        speaker_id=0,
    ),
    valid=dict(
        type="NaiveSVCDataset",
        path="dataset/valid",
        speaker_id=0,
    ),
)

dataloader = dict(
    train=dict(
        batch_size=40,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)
