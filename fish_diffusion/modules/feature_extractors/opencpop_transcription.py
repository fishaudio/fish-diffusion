from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class OpenCpopTranscriptionToPhonemesDuration(BaseFeatureExtractor):
    def __init__(self, phonemes: list[str], transcription_path: str):
        super().__init__()

        self.phonemes = phonemes
        self.transcription_path = transcription_path

        self.transcriptions = self._load_transcriptions(transcription_path)

    def _load_transcriptions(self, transcription_path: str):
        results = {}

        for i in open(transcription_path):
            id, _, phones, _, _, durations, _ = i.split("|")
            phones = phones.split(" ")
            durations = [float(i) for i in durations.split(" ")]

            assert len(phones) == len(durations)

            results[id] = (phones, durations)

        return results

    @torch.no_grad()
    def forward(self, audio_path: Path, mel_len: int):
        id = audio_path.stem
        phones, durations = self.transcriptions[id]

        cumsum_durations = np.cumsum(durations)
        alignment_factor = mel_len / cumsum_durations[-1]

        # Create one-hot encoding for phonemes
        features = F.one_hot(
            torch.tensor([self.phonemes.index(i) for i in phones]),
            num_classes=len(self.phonemes),
        ).float()

        # Create phones to mel alignment
        phones2mel = torch.zeros(mel_len, dtype=torch.long)

        for i, sum_duration in enumerate(cumsum_durations):
            current_idx = int(sum_duration * alignment_factor)
            previous_idx = (
                int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
            )
            phones2mel[previous_idx:current_idx] = i

        return features.T, phones2mel
