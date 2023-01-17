import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class Wav2Vec2XLSR(BaseFeatureExtractor):
    def __init__(self, name="voidful/wav2vec2-xlsr-multilingual-56"):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name)
        self.model.eval()

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        outputs = self.model(input_values, output_hidden_states=True)

        # (B, D, T) -> (B, T, D)
        return outputs.hidden_states[-1].transpose(1, 2)


@FEATURE_EXTRACTORS.register_module()
class Wav2Vec2XLSRIPA(Wav2Vec2XLSR):
    def __init__(self, name="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        super().__init__(name)

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        logits = self.model(input_values).logits
        logits = logits.softmax(-1)

        # Mask out everything < 0.1
        logits = logits.masked_fill(logits < 0.1, 0.0)

        # (B, D, T) -> (B, T, D)
        return logits.transpose(1, 2)
