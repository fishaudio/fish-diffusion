from typing import Optional

import torch
from torch import nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class ChineseHubertSoft(BaseFeatureExtractor):
    def __init__(self, pretrained: bool = True, checkpoint_path: Optional[str] = None):
        super().__init__()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "TencentGameMate/chinese-hubert-base"
        )
        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 256))

        state_dict = None

        if pretrained is True:
            state_dict = torch.hub.load_state_dict_from_url(
                "https://github.com/fishaudio/chinese-hubert-soft/releases/download/v1/chinese-hubert-soft-v1.ckpt",
                map_location="cpu",
            )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)

        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # Top-k gating
        topk, indices = torch.topk(features, 10, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)

        return features.transpose(1, 2)
