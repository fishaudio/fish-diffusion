from typing import Optional

import torch
from transformers import AutoProcessor, EncodecModel

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


class EncodecOverrideModel(EncodecModel):
    def _decode_frame(
        self, codes: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        return self.quantizer.decode(codes)


@FEATURE_EXTRACTORS.register_module()
class Encodec(BaseFeatureExtractor):
    sampling_rate = 24000

    def __init__(
        self,
        model: str = "facebook/encodec_24khz",
        bandwidth: float = 1.5,
        first_codebook_only: bool = False,
    ):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model)
        self.model = EncodecOverrideModel.from_pretrained(model)
        self.model.eval()

        self.bandwidth = bandwidth
        self.first_codebook_only = first_codebook_only

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audios = self.preprocess(path_or_audio, sampling_rate)
        return self._forward(audios)

    @torch.no_grad()
    def _forward(self, audio):
        if audio.dim() == 1:
            audio = audio[None]

        audio = audio.to(self.device)
        assert audio.dim() == 2

        x = self.model.encode(
            audio[:, None], bandwidth=self.bandwidth, return_dict=True
        )
        if self.first_codebook_only:
            x.audio_codes = x.audio_codes[:, :, :1, :]

        return self.model.decode(x.audio_codes, x.audio_scales).audio_values
