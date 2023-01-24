from pathlib import Path

import numpy as np
import torch

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
            durations = durations.split(" ")

            assert len(phones) == len(durations)

            results[id] = (phones, durations)

        return results

    @torch.no_grad()
    def forward(self, audio_path: Path):
        id = audio_path.stem
        phones, durations = self.transcriptions[id]

        features = torch.zeros(
            (len(phones), len(self.phonemes) + 1), dtype=torch.float32
        )

        for i, (phone, duration) in enumerate(zip(phones, durations)):
            features[i, self.phonemes.index(phone)] = 1.0
            features[i, -1] = float(duration)

        return features
