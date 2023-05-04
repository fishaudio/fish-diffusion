import numpy as np
import torch
from torch.nn import functional as F

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register_module()
class PlainTextExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, text, path_or_audio=None, sampling_rate=None):
        self.preprocess(path_or_audio, sampling_rate)
        return self._forward(text)

    @torch.no_grad()
    def _forward(self, text):
        text = text[None].to(self.device)
        text = self.env.encode(text)
        return text
