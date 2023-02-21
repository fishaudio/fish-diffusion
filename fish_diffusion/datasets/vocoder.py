import json
import math
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
from fish_audio_preprocess.utils.file import list_files
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class VOCODERDataset(Dataset):
    def __init__(self, path="dataset", segment_size=16384, hop_size=512):
        self.wav_paths = list_files(path, {".wav"}, recursive=True)
        self.dataset_path = Path(path)

        self.segment_size = segment_size
        self.hop_size = hop_size

        assert (
            len(self.wav_paths) > 0
        ), f"No wav files found in {path}, please check your path."

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path: Path = self.wav_paths[idx]
        wav_path = str(wav_path)

        mel_path = wav_path + ".mel.npy"
        mel = np.load(mel_path)

        pitch_path = wav_path + ".f0.npy"
        pitch = np.load(pitch_path)
        audio, _ = librosa.load(wav_path, sr=44100, mono=True)
        audio = audio[None]

        # random segment
        if mel.shape[1] > self.segment_size // self.hop_size:
            start = np.random.randint(0, audio.shape[1] - self.segment_size + 1)
            audio = audio[:, start : start + self.segment_size]
            mel = mel[
                :, start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]
            pitch = pitch[
                start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]
        else:
            audio = np.pad(
                audio, ((0, 0), (0, self.segment_size - audio.shape[1])), "constant"
            )
            print(self.segment_size // self.hop_size - mel.shape[1])
            mel = np.pad(
                mel,
                ((0, 0), (0, self.segment_size // self.hop_size - mel.shape[1])),
                "constant",
            )
            pitch = np.pad(
                pitch,
                ((0, self.segment_size // self.hop_size - pitch.shape[0])),
                "constant",
            )

        sample = {
            "audios": audio,
            "mels": mel,
            "pitches": pitch,
        }

        return sample
