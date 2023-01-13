import torch.nn as nn
from diff_svc.denoisers.wavenet import WaveNetDenoiser

import utils.pitch_tools
from utils.tools import get_mask_from_lengths

from diff_svc.encoders import ENCODERS
from diff_svc.diffusions import DIFFUSIONS


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)
        self.pitch_encoder = ENCODERS.build(model_config.pitch_encoder)

    def forward_features(
        self,
        speakers,
        contents,
        src_lens,
        max_src_len,
        mel_lens=None,
        max_mel_len=None,
        pitches=None,
    ):
        src_masks = (
            get_mask_from_lengths(src_lens, max_src_len)
            if src_lens is not None
            else None
        )

        speaker_embed = self.speaker_encoder(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
        features = self.text_encoder(contents, src_masks, speaker_embed)
        features += self.pitch_encoder(utils.pitch_tools.f0_to_coarse(pitches))

        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        return dict(
            features=features,
            src_masks=src_masks,
            mel_masks=mel_masks,
        )

    def forward(
        self,
        speakers,
        contents,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        pitches=None,
    ):
        features = self.forward_features(
            speakers=speakers,
            contents=contents,
            src_lens=src_lens,
            max_src_len=max_src_len,
            mel_lens=mel_lens,
            max_mel_len=max_mel_len,
            pitches=pitches,
        )

        output_dict = self.diffusion(features["features"], mels, features["mel_masks"])

        # For validation
        output_dict["features"] = features["features"]

        return output_dict
