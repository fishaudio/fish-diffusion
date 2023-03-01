from .audio_folder import AudioFolderDataset
from .builder import DATASETS
from .concat import ConcatDataset
from .naive import NaiveDataset, NaiveSVCDataset
from .repeat import RepeatDataset
from .vocoder import VOCODERDataset

__all__ = [
    "DATASETS",
    "AudioFolderDataset",
    "ConcatDataset",
    "RepeatDataset",
    "VOCODERDataset",
    "NaiveDataset",
    "NaiveSVCDataset",
]
