import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class Wav2Vec2(BaseFeatureExtractor):
    def __init__(
        self,
        checkpoint_path: str = "facebook/mms-1b",
    ):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            checkpoint_path
        )
        self.model = Wav2Vec2ForPreTraining.from_pretrained(checkpoint_path)

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(self.model.device)

        return self._forward(input_values)

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = features.projected_quantized_states

        return features.transpose(1, 2)
