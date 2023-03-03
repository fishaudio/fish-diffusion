from pathlib import Path

import numpy as np
import torch
from fish_audio_preprocess.utils.file import list_files
from torch.utils.data import Dataset

from fish_diffusion.datasets.utils import transform_pipeline

from .builder import DATASETS


@DATASETS.register_module()
class NaiveDataset(Dataset):
    processing_pipeline = []

    collating_pipeline = []

    def __init__(self, path="dataset", speaker_id=0):
        self.paths = list_files(path, {".npy"}, recursive=True, sort=True)
        self.dataset_path = Path(path)
        self.speaker_id = speaker_id

        assert len(self.paths) > 0, f"No files found in {path}, please check your path."

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx], allow_pickle=True).item()
        x["speaker"] = self.speaker_id

        return transform_pipeline(self.processing_pipeline, x)

    @classmethod
    def collate_fn(cls, data):
        return transform_pipeline(cls.collating_pipeline, data)


@DATASETS.register_module()
class NaiveSVCDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=["path", "time_stretch", "mel", "contents", "pitches", "key_shift", "speaker"],
        ),
        dict(type="Transpose", keys=[("mel", 1, 0), ("contents", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("mel", -2), ("contents", -2), ("pitches", -1)]),
        dict(
            type="ToTensor",
            keys=[
                ("time_stretch", torch.float32),
                ("key_shift", torch.float32),
                ("speaker", torch.int64),
            ],
        ),
        dict(type="UnSqueeze", keys=[("pitches", -1), ("time_stretch", -1), ("key_shift", -1)]), # (N, T) -> (N, T, 1)
    ]


@DATASETS.register_module()
class NaiveVOCODERDataset(NaiveDataset):
    processing_pipeline = [
        dict(type="PickKeys", keys=["path", "audio", "mel", "pitches", "key_shift"]),
        dict(type="Transpose", keys=[("mel", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("audio", -1), ("mel", -2), ("pitches", -1)]),
        dict(
            type="ToTensor",
            keys=[("key_shift", torch.float32)],
        ),
    ]

    def __init__(
        self, path="dataset", segment_size=16384, hop_size=512, sampling_rate=44100
    ):
        super().__init__(path)

        self.segment_size = segment_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate

    def __getitem__(self, idx):
        x = super().__getitem__(idx)

        # Randomly crop the audio and mel
        if (
            self.segment_size is not None
            and self.segment_size > 0
            and x["mel"].shape[1] > self.segment_size // self.hop_size
        ):
            start = np.random.randint(0, x["audio"].shape[1] - self.segment_size + 1)
            x["audio"] = x["audio"][:, start : start + self.segment_size]
            x["mel"] = x["mel"][
                :, start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]
            x["pitches"] = x["pitches"][
                start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]

        return x
