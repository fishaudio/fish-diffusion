dataset = dict(
    train=dict(
        type="HiFiSVCDataset",
        path="dataset/train",
        speaker_id=0,
        segment_size=16384,
    ),
    valid=dict(
        type="HiFiSVCDataset",
        path="dataset/valid",
        speaker_id=0,
        segment_size=-1,
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
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)
