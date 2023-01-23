import torch
from torch import nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
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

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        features = self.model(input_values).last_hidden_state

        if self.discrete is False:
            # (B, D, T) -> (B, T, D)
            return features.transpose(1, 2)

        dist = torch.cdist(features, self.cluster_centers)

        return dist.log_softmax(2).transpose(1, 2)


@FEATURE_EXTRACTORS.register_module()
class ChineseHubertSoft(BaseFeatureExtractor):
    def __init__(self, ckpt_path=None):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "TencentGameMate/chinese-hubert-base"
        )
        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 256))

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # features = features.softmax(2)
        # features[features < 0.1] = 0

        return features.transpose(1, 2)
