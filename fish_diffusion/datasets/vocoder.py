import json
import math
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
import torch
from fish_audio_preprocess.utils.file import list_files
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class VOCODERDataset(Dataset):
    def __init__(
        self, path="dataset", segment_size=16384, hop_size=512, sampling_rate=44100
    ):
        self.wav_paths = list_files(path, {".wav"}, recursive=True)
        self.dataset_path = Path(path)

        self.segment_size = segment_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate

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
        audio, _ = librosa.load(wav_path, sr=self.sampling_rate, mono=True)
        audio = audio[None]

        # Randomly crop the audio and mel
        if (
            self.segment_size is not None
            and self.segment_size > 0
            and mel.shape[1] > self.segment_size // self.hop_size
        ):
            start = np.random.randint(0, audio.shape[1] - self.segment_size + 1)
            audio = audio[:, start : start + self.segment_size]
            mel = mel[
                :, start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]
            pitch = pitch[
                start // self.hop_size : (start + self.segment_size) // self.hop_size
            ]

        return {
            "audio": audio,
            "mel": mel,
            "pitch": pitch,
        }

    @staticmethod
    def collate_fn(data):
        info = {}

        # Create audios and padding
        audios = [torch.from_numpy(d["audio"]).float() for d in data]
        audio_lens = torch.LongTensor([audio.shape[1] for audio in audios])
        max_audio_len = torch.max(audio_lens)
        audios = torch.stack(
            [
                torch.nn.functional.pad(audio, (0, max_audio_len - audio.shape[1]))
                for audio in audios
            ]
        )

        info["audios"] = audios
        info["audio_lens"] = audio_lens
        info["max_audio_len"] = max_audio_len

        # Create mels and padding
        mels = [torch.from_numpy(d["mel"]).float() for d in data]
        mel_lens = torch.LongTensor([mel.shape[1] for mel in mels])
        max_mel_len = torch.max(mel_lens)
        mels = torch.stack(
            [
                torch.nn.functional.pad(mel, (0, max_mel_len - mel.shape[1]))
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
