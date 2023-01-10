import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from .base import BaseFeatureExtractor


class Wav2Vec2XLSR(BaseFeatureExtractor):
    def __init__(self, name="voidful/wav2vec2-xlsr-multilingual-56"):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name)
        self.model.eval()

    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        return outputs.hidden_states[-1]
