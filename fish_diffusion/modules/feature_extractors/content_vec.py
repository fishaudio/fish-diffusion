import torch
from torch import nn
from transformers import HubertModel

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


@FEATURE_EXTRACTORS.register_module()
class ContentVec(BaseFeatureExtractor):
    def __init__(
        self,
        checkpoint_path: str = "lengyue233/content-vec-best",
        output_layer: int = 9,
        use_projection: bool = True,
    ):
        super().__init__()

        self.model = HubertModelWithFinalProj.from_pretrained(checkpoint_path)
        self.model.eval()

        self.output_layer = output_layer
        self.use_projection = use_projection

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)
        return self._forward(audio)

    @torch.no_grad()
    def _forward(self, audio):
        if audio.dim() == 1:
            audio = audio[None]

        audio = audio.to(self.device)
        assert audio.dim() == 2

        if self.output_layer is not None and self.output_layer >= 0:
            x = self.model(audio, output_hidden_states=True)["hidden_states"][
                self.output_layer
            ]
        else:
            x = self.model(audio)["last_hidden_state"]

        if self.use_projection:
            x = self.model.final_proj(x)

        return x.transpose(1, 2)
