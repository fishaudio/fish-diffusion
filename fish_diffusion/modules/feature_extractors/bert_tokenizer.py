from pathlib import Path

import torch
from transformers import AutoTokenizer

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class BertTokenizer(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        transcription_path: str,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transcription_path = Path(transcription_path)

        self.transcriptions = self._load_transcriptions(transcription_path)

    def _load_transcriptions(self, transcription_path: str):
        results = {}

        for i in open(transcription_path):
            id, text = i.split("|")

            results[id] = text.strip()

        return results

    @torch.no_grad()
    def forward(self, audio_path: Path):
        id = str(
            audio_path.absolute().relative_to(self.transcription_path.parent.absolute())
        )
        text = self.transcriptions[id]

        data = self.tokenizer.encode_plus(text, return_offsets_mapping=True)

        input_ids = data["input_ids"]
        offset_mapping = data["offset_mapping"]

        # Aligning input_ids with offset_mapping
        new_input_ids = []
        for input_id, (start, end) in zip(input_ids, offset_mapping):
            length = end - start
            new_input_ids.extend(
                [input_id] + [self.tokenizer.pad_token_id] * (length - 1)
            )

        # Adding <pad> between each word
        input_ids = torch.tensor(new_input_ids, dtype=torch.long)
        new_input_ids = torch.tensor(
            [self.tokenizer.pad_token_id] * (len(input_ids) * 2 - 1), dtype=torch.long
        )
        new_input_ids[::2] = input_ids

        return new_input_ids
