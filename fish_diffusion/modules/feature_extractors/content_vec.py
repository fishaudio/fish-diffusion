import logging
import os

import torch
from loguru import logger

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS

# Ignore fairseq's logger
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)

from fairseq import checkpoint_utils


@FEATURE_EXTRACTORS.register_module()
class ContentVec(BaseFeatureExtractor):
    def __init__(
        self, checkpoint_path: str = "checkpoints/content-vec-best-legacy-500.pt"
    ):
        super().__init__()

        if os.path.exists(checkpoint_path) is False:
            logger.error(f"Checkpoint {checkpoint_path} does not exist")
            logger.error(
                f"If you are trying to use the pretrained ContentVec, you can either:"
            )
            logger.error(
                f"1. Download the ContentVec from https://github.com/fishaudio/fish-diffusion/releases/tag/v1.12"
            )
            logger.error(
                f"2. Run `python tools/download_nsf_hifigan.py --content-vec` to download the ContentVec model"
            )

            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], suffix=""
        )
        self.model = models[0]
        self.model.eval()

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)
        return self._forward(audio)

    @torch.no_grad()
    def _forward(self, audio):
        audio = audio[None].to(self.device)
        assert audio.dim() == 2

        padding_mask = torch.zeros(audio.shape, dtype=torch.bool, device=audio.device)
        inputs = {"source": audio, "padding_mask": padding_mask, "output_layer": 9}

        features = self.model.extract_features(**inputs)
        units = self.model.final_proj(features[0])

        return units.transpose(1, 2)
