import torch

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from .base import BaseFeatureExtractor


class ChineseHubert(BaseFeatureExtractor):
    def __init__(self, name="TencentGameMate/chinese-hubert-large"):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = HubertModel.from_pretrained(name)
        self.model.eval()

    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_values)
        
        # (B, D, T) -> (B, T, D)
        return outputs.last_hidden_state.transpose(1, 2)
