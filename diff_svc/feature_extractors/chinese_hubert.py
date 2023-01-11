import torch

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

from .base import BaseFeatureExtractor


class ChineseHubert(BaseFeatureExtractor):
    def __init__(self, name="TencentGameMate/chinese-hubert-base", discrete=False):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = HubertModel.from_pretrained(name)
        self.model.eval()

        self.discrete = discrete

        if discrete:
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://github.com/bshall/hubert/releases/download/v0.1/kmeans100-50f36a95.pt",
                progress=True,
            )
            self.cluster_centers = torch.nn.Parameter(checkpoint["cluster_centers_"])

    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        features = outputs.last_hidden_state

        if self.discrete is False:
            # (B, D, T) -> (B, T, D)
            return features.transpose(1, 2)

        dist = torch.cdist(features, self.cluster_centers)

        return dist.log_softmax(2).transpose(1, 2)
