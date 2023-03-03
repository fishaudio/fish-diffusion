from fish_diffusion.datasets.naive import NaiveSVCDataset

_base_ = [
    "./svc_hubert_soft.py",
]

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="NaiveSVCDataset",
                path="dataset/speaker_0",
                speaker_id=0,
            ),
            dict(
                type="NaiveSVCDataset",
                path="dataset/speaker_1",
                speaker_id=1,
            ),
        ],
        # Are there any other ways to do this?
        collate_fn=NaiveSVCDataset.collate_fn,
    ),
    valid=dict(
        type="NaiveSVCDataset",
        path="dataset/valid",
        speaker_id=0,
    ),
)

model = dict(
    speaker_encoder=dict(
        input_size=2,  # 2 speakers
    ),
)
