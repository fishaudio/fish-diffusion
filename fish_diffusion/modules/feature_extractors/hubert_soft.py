import torch

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class HubertSoft(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("bshall/hubert:main", "hubert_soft")

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)
        return self._forward(audio[None])

    @torch.no_grad()
    def _forward(self, audio):
        audio = audio[None].to(self.device)
        units = self.model.units(audio)

        return units.transpose(1, 2)
