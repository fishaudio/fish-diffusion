import numpy as np
import torch

from fish_diffusion.datasets.builder import DATASETS
from fish_diffusion.datasets.naive import NaiveDataset


@DATASETS.register_module()
class HiFiSVCDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "time_stretch",
                "audio",
                "contents",
                "pitches",
                "key_shift",
                "speaker",
            ],
        ),
        dict(type="UnSqueeze", keys=[("audio", 0)]),
        dict(type="Transpose", keys=[("contents", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("audio", -1), ("contents", -2), ("pitches", -1)]),
        dict(
            type="ToTensor",
            keys=[
                ("time_stretch", torch.float32),
                ("key_shift", torch.float32),
                ("speaker", torch.int64),
            ],
        ),
        dict(
            type="UnSqueeze",
            keys=[("pitches", -1), ("time_stretch", -1), ("key_shift", -1)],
        ),  # (N, T) -> (N, T, 1)
    ]

    def __init__(self, path="dataset", speaker_id=0, segment_size=-1, hop_length=512):
        super().__init__(path, speaker_id)

        self.segment_size = segment_size
        self.hop_length = hop_length

    def __getitem__(self, idx):
        x = super().__getitem__(idx)

        # Randomly crop the audio and mel
        if (
            self.segment_size is not None
            and self.segment_size > 0
            and x["contents"].shape[1] > self.segment_size // self.hop_length
            and x["audio"].shape[1] > self.segment_size
        ):
            mel_crop = lambda x: x[
                start
                // self.hop_length : (start + self.segment_size)
                // self.hop_length
            ]
            start = np.random.randint(0, x["audio"].shape[1] - self.segment_size + 1)
            x["audio"] = x["audio"][:, start : start + self.segment_size]
            x["pitches"] = mel_crop(x["pitches"])
            x["contents"] = mel_crop(x["contents"])

        return x
