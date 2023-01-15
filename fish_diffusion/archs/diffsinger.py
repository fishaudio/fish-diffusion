import torch.nn as nn

import torch

from fish_diffusion.encoders import ENCODERS
from fish_diffusion.diffusions import DIFFUSIONS
from fish_diffusion.utils.pitch import f0_to_coarse


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)
        self.pitch_encoder = ENCODERS.build(model_config.pitch_encoder)

    @staticmethod
    def get_mask_from_lengths(lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

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
            self.get_mask_from_lengths(src_lens, max_src_len)
            if src_lens is not None
            else None
        )

        speaker_embed = (
            self.speaker_encoder(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
        )
        features = self.text_encoder(contents, src_masks, speaker_embed)
        features += self.pitch_encoder(f0_to_coarse(pitches))

        mel_masks = (
            self.get_mask_from_lengths(mel_lens, max_mel_len)
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
