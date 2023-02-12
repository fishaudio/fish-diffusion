import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from fish_audio_preprocess.utils.file import list_files
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class AudioFolderDataset(Dataset):
    def __init__(self, path="dataset", speaker_id=0):
        self.wav_paths = list_files(path, {".wav"}, recursive=True)
        self.dataset_path = Path(path)
        self.speaker_id = speaker_id

        assert (
            len(self.wav_paths) > 0
        ), f"No wav files found in {path}, please check your path."

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path: Path = self.wav_paths[idx]

        wav_path = str(wav_path)

        mel_path = wav_path + ".mel.npy"
        mel = np.load(mel_path).T

        c_path = wav_path + f".text_features.npy"
        c = np.load(c_path).T
        pitch_path = wav_path + ".f0.npy"
        pitch = np.load(pitch_path)

        sample = {
            "path": wav_path,
            "speaker": self.speaker_id,
            "content": c,
            "mel": mel,
            "pitch": pitch,
        }

        return sample

    @staticmethod
    def collate_fn(data):
        info = {
            "paths": [d["path"] for d in data],
            "speakers": torch.LongTensor([d["speaker"] for d in data]),
        }

        # Create contents and padding
        contents = [torch.from_numpy(d["content"]).float() for d in data]
        content_lens = torch.LongTensor([c.shape[0] for c in contents])
        max_content_len = torch.max(content_lens)
        contents = torch.stack(
            [
                torch.nn.functional.pad(c, (0, 0, 0, max_content_len - c.shape[0]))
                for c in contents
            ]
        )

        info["contents"] = contents
        info["content_lens"] = content_lens
        info["max_content_len"] = max_content_len

        # Create mels and padding
        mels = [torch.from_numpy(d["mel"]).float() for d in data]
        mel_lens = torch.LongTensor([mel.shape[0] for mel in mels])
        max_mel_len = torch.max(mel_lens)
        mels = torch.stack(
            [
                torch.nn.functional.pad(mel, (0, 0, 0, max_mel_len - mel.shape[0]))
                for mel in mels
            ]
        )

        info["mels"] = mels
        info["mel_lens"] = mel_lens
        info["max_mel_len"] = max_mel_len

        # Create pitches and padding
        pitches = [torch.from_numpy(d["pitch"]).float() for d in data]
        pitch_lens = torch.LongTensor([pitch.shape[0] for pitch in pitches])
        max_pitch_len = torch.max(pitch_lens)
        pitches = torch.stack(
            [
                torch.nn.functional.pad(pitch, (0, max_pitch_len - pitch.shape[0]))
                for pitch in pitches
            ]
        )

        info["pitches"] = pitches
        info["pitch_lens"] = pitch_lens
        info["max_pitch_len"] = max_pitch_len

        return info

    @staticmethod
    def get_speaker_map_from_subfolder(path, existing_speaker_map=None):
        if existing_speaker_map is None:
            speaker_map = {}
        else:
            speaker_map = deepcopy(existing_speaker_map)

        for speaker_path in sorted(Path(path).iterdir()):
            if not speaker_path.is_dir() or speaker_path.name.startswith("."):
                continue

            speaker_map[str(speaker_path.name)] = len(speaker_map)

        return speaker_map

    @staticmethod
    def get_datasets_from_subfolder(
        path, speaker_map: dict[str, int]
    ) -> list["AudioFolderDataset"]:
        datasets = []

        for speaker_path in sorted(Path(path).iterdir()):
            if not speaker_path.is_dir() or speaker_path.name.startswith("."):
                continue

            speaker_id = speaker_map[str(speaker_path.name)]
            datasets.append(
                dict(
                    type="AudioFolderDataset",
                    path=str(speaker_path),
                    speaker_id=speaker_id,
                )
            )

        return datasets
