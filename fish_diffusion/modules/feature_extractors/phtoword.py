from pathlib import Path

import librosa
import numpy as np
import csv
import torch
from torch.nn import functional as F

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class PhoneToWord(BaseFeatureExtractor):
    def __init__(self, phonemes: list[str], transcription_path: str):
        super().__init__()

        self.phonemes = phonemes
        self.transcription_path = transcription_path

        self.transcriptions = self._load_transcriptions(transcription_path)

    def _load_transcriptions(self, transcription_path: str):
        results = {}

        # name,ph_seq,ph_dur,ph_num,note_seq,note_dur
        for i in csv.DictReader(open(transcription_path)):
            id = i["name"]
            phones = i["ph_seq"].split(" ")
            phone_durations = [float(i) for i in i["ph_dur"].split(" ")]
            notes = i["note_seq"].split(" ")
            note_durations = [float(i) for i in i["note_dur"].split(" ")]

            assert len(phones) == len(phone_durations)
            assert len(notes) == len(note_durations)

            results[id] = {
                "phones": phones,
                "phone_durations": phone_durations,
                "notes": notes,
                "note_durations": note_durations,
            }

        return results

    @torch.no_grad()
    def forward(self, audio_path: Path, mel_len: int):
        id = audio_path.stem
        phones = self.transcriptions[id]["phones"]
        durations = self.transcriptions[id]["phone_durations"]

        # Create one-hot encoding for phonemes
        features = F.one_hot(
            torch.tensor([self.phonemes.index(i) for i in phones]),
            num_classes=len(self.phonemes),
        ).float()

        # Create phones to mel alignment
        cumsum_durations = np.cumsum(durations)
        alignment_factor = mel_len / cumsum_durations[-1]

        phones2mel = torch.zeros(mel_len, dtype=torch.long)

        for i, sum_duration in enumerate(cumsum_durations):
            current_idx = int(sum_duration * alignment_factor)
            previous_idx = (
                int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
            )
            phones2mel[previous_idx:current_idx] = i

        return features.T, phones2mel

    def notes_f0(self, audio_path: Path, mel_len: int):
        id = audio_path.stem
        notes = self.transcriptions[id]["notes"]
        durations = self.transcriptions[id]["note_durations"]

        f0s = []
        for note in notes:
            if note == "rest":
                f0s.append(0)
                continue

            if "/" in note:
                note, _ = note.split("/")

            f0 = librosa.note_to_hz(note)
            f0s.append(f0)

        pitches = torch.tensor(f0s, dtype=torch.float)

        # Create phones to mel alignment
        cumsum_durations = np.cumsum(durations)
        alignment_factor = mel_len / cumsum_durations[-1]

        pitches2mel = torch.zeros(mel_len, dtype=torch.long)

        for i, sum_duration in enumerate(cumsum_durations):
            current_idx = int(sum_duration * alignment_factor)
            previous_idx = (
                int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
            )
            pitches2mel[previous_idx:current_idx] = i
        
        return pitches, pitches2mel
