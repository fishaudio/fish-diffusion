
from fish_diffusion.datasets.naive import NaiveTTSDataset


_base_ = [
    "./tts_diff_base.py",
]


preprocessing = dict(
    text_features_extractor=dict(
        type="",
    ),
    pitch_extractor=dict(
        # ParselMouth is much faster than Crepe
        # However, Crepe may have better performance in some cases
        type="ParselMouthPitchExtractor",
        keep_zeros=False,
    ),
)

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
        collate_fn=NaiveTTSDataset.collate_fn,
    ),
    valid=dict(
        type="NaiveSVCDataset",
        path="dataset/valid",
        speaker_id=0,
    ),
)