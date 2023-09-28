from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
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

    def get_item(self, idx):
        x = np.load(self.paths[idx], allow_pickle=True).item()
        x["speaker"] = self.speaker_id

        return transform_pipeline(self.processing_pipeline, x)

    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except Exception:
            print(f"Error when loading {self.paths[idx]}, skipping...")
            return None

    @classmethod
    def collate_fn(cls, data):
        # Remove None
        data = [x for x in data if x is not None]

        return transform_pipeline(cls.collating_pipeline, data)


@DATASETS.register_module()
class NaiveSVCDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "time_stretch",
                "mel",
                "contents",
                "pitches",
                "key_shift",
                "speaker",
            ],
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
        dict(
            type="UnSqueeze",
            keys=[("pitches", -1), ("time_stretch", -1), ("key_shift", -1)],
        ),  # (N, T) -> (N, T, 1)
    ]


@DATASETS.register_module()
class NaiveSVCPowerDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "time_stretch",
                "mel",
                "contents",
                "pitches",
                "key_shift",
                "speaker",
                "energy",
            ],
        ),
        dict(type="Transpose", keys=[("mel", 1, 0), ("contents", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(
            type="PadStack",
            keys=[("mel", -2), ("contents", -2), ("pitches", -1), ("energy", -1)],
        ),
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
            keys=[
                ("pitches", -1),
                ("time_stretch", -1),
                ("key_shift", -1),
                ("energy", -1),
            ],
        ),  # (N, T) -> (N, T, 1)
    ]


@DATASETS.register_module()
class NaiveVOCODERDataset(NaiveDataset):
    processing_pipeline = [
        dict(type="PickKeys", keys=["path", "audio", "pitches", "sampling_rate"]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("audio", -1), ("pitches", -1)]),
    ]

    def __init__(
        self,
        path="dataset",
        segment_size: Optional[int] = 16384,
        hop_length: int = 512,
        sampling_rate: int = 44100,
        pitch_shift: Optional[list[int]] = None,
        loudness_shift: Optional[list[int]] = None,
    ):
        super().__init__(path)

        self.segment_length = segment_size
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.pitch_shift = pitch_shift
        self.loudness_shift = loudness_shift

    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        assert x["sampling_rate"] == self.sampling_rate

        y = x["audio"]
        pitches = x["pitches"]

        if self.pitch_shift is not None:
            shift = (
                np.random.random() * (self.pitch_shift[1] - self.pitch_shift[0])
                + self.pitch_shift[0]
            )
            duration_shift = 2 ** (shift / 12)
            orig_sr = round(self.sampling_rate * duration_shift)
            orig_sr = orig_sr - (orig_sr % 100)

            y = torchaudio.functional.resample(
                torch.from_numpy(y).float(),
                orig_freq=orig_sr,
                new_freq=self.sampling_rate,
            ).numpy()

            # Adjust pitch
            pitches *= 2 ** (shift / 12)

        pitches = np.interp(
            np.linspace(0, 1, y.shape[-1]), np.linspace(0, 1, len(pitches)), pitches
        )

        if self.segment_length is not None and y.shape[-1] > self.segment_length:
            start = np.random.randint(0, y.shape[-1] - self.segment_length + 1)
            y = y[start : start + self.segment_length]
            pitches = pitches[start : start + self.segment_length]

        if self.loudness_shift is not None:
            new_amplitude = (
                np.random.random() * (self.loudness_shift[1] - self.loudness_shift[0])
                + self.loudness_shift[0]
            )
            max_amplitude = np.max(np.abs(y))
            y = y / (max_amplitude + 1e-8) * new_amplitude

        return {
            "audio": y[None],
            "pitches": pitches[None],
        }


@DATASETS.register_module()
class NaiveSVSDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "time_stretch",
                "mel",
                "contents",
                "pitches",
                "key_shift",
                "speaker",
                # Add a phones2mel key compared to NaiveSVCDataset
                # This is used for the phoneme-based fs2 model
                "phones2mel",
            ],
        ),
        dict(type="Transpose", keys=[("mel", 1, 0), ("contents", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(
            type="PadStack",
            keys=[("mel", -2), ("contents", -2), ("pitches", -1), ("phones2mel", -1)],
        ),
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
        ),
    ]


@DATASETS.register_module()
class NaiveTTSDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "mel",
                "contents",
                "speaker",
            ],
        ),
        dict(type="Transpose", keys=[("mel", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="FilterByLength", key="mel", dim=0, min_length=1, max_length=2048),
        dict(type="ListToDict"),
        dict(
            type="PadStack",
            keys=[("mel", -2), ("contents", -1)],
        ),
        dict(
            type="ToTensor",
            keys=[
                ("speaker", torch.int64),
                ("contents", torch.int64),
            ],
        ),
    ]


@DATASETS.register_module()
class NaiveDenoiserDataset(NaiveDataset):
    processing_pipeline = [
        dict(
            type="PickKeys",
            keys=[
                "path",
                "mel",
                "contents",
            ],
        ),
        dict(type="Transpose", keys=[("mel", 1, 0), ("contents", 1, 0)]),
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("mel", -2), ("contents", -2)]),
    ]
