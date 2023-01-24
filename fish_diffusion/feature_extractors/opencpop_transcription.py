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

        features = torch.zeros((mel_len, len(self.phonemes) + 1), dtype=torch.float32)

        for i, (phone, duration, sum_duration) in enumerate(
            zip(phones, durations, cumsum_durations)
        ):
            current_idx = int(sum_duration * alignment_factor)
            previous_idx = (
                int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
            )
            _temp = torch.zeros(len(self.phonemes) + 1, dtype=torch.float32)
            _temp[self.phonemes.index(phone)] = 1
            _temp[-1] = duration

            features[previous_idx:current_idx] = _temp

        return features.T
