import torch
import torch.nn as nn

from fish_diffusion.modules.encoders import ENCODERS

from .diffusions import DIFFUSIONS


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)
        self.pitch_encoder = ENCODERS.build(model_config.pitch_encoder)

        if "pitch_shift_encoder" in model_config:
            self.pitch_shift_encoder = ENCODERS.build(model_config.pitch_shift_encoder)

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
        contents_lens,
        contents_max_len,
        mel_lens=None,
        mel_max_len=None,
        pitches=None,
        pitch_shift=None,
    ):
        src_masks = (
            self.get_mask_from_lengths(contents_lens, contents_max_len)
            if contents_lens is not None
            else None
        )

        features = self.text_encoder(contents, src_masks)

        speaker_embed = (
            self.speaker_encoder(speakers).unsqueeze(1).expand(-1, contents_max_len, -1)
        )

        features += speaker_embed
        features += self.pitch_encoder(pitches)

        if pitch_shift is not None and hasattr(self, "pitch_shift_encoder"):
            features += self.pitch_shift_encoder(pitch_shift)[:, None]

        mel_masks = (
            self.get_mask_from_lengths(mel_lens, mel_max_len)
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
        contents_lens,
        contents_max_len,
        mel=None,
        mel_lens=None,
        mel_max_len=None,
        pitches=None,
        pitch_shift=None,
    ):
        features = self.forward_features(
            speakers=speakers,
            contents=contents,
            contents_lens=contents_lens,
            contents_max_len=contents_max_len,
            mel_lens=mel_lens,
            mel_max_len=mel_max_len,
            pitches=pitches,
            pitch_shift=pitch_shift,
        )

        output_dict = self.diffusion.train_step(
            features["features"], mel, features["mel_masks"]
        )

        # For validation
        output_dict["features"] = features["features"]

        return output_dict
