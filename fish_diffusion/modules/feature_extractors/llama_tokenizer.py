from pathlib import Path

import torch
from transformers import AutoTokenizer

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class LlamaTokenizer(BaseFeatureExtractor):
    def __init__(
        self,
        model_name: str,
        label_suffix: str = ".txt",
        speaker_mode: str = "libritts",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_suffix = label_suffix
        self.speaker_mode = speaker_mode

    @torch.no_grad()
    def forward(self, audio_path: Path):
        transcript = audio_path.with_suffix(self.label_suffix).read_text().strip()
        speaker = audio_path.parent.parent.name

        transcript = f"[spk] {speaker} [txt] {transcript} [mel]"
        input_ids = self.tokenizer.encode(transcript, return_tensors="pt")

        return input_ids
