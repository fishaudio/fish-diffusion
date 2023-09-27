import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fish_diffusion.modules.encoders import ENCODERS

from .diffusions import DIFFUSIONS


class GradTTS(nn.Module):
    """GradTTS"""

    def __init__(self, model_config):
        super(GradTTS, self).__init__()

        self.diffusion = DIFFUSIONS.build(model_config.diffusion)

        if getattr(model_config, "gradient_checkpointing", False):
            self.diffusion.denoise_fn.gradient_checkpointing_enable()

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
        mel=None,
        mel_lens=None,
        mel_max_len=None,
        pitches=None,
        pitch_shift=None,
        phones2mel=None,
        energy=None,
    ):
        src_masks = self.get_mask_from_lengths(contents_lens, contents_max_len)

        if self.training is False:
            # Random shift mel_lens by 10%
            mel_lens = mel_lens * (0.9 + 0.2 * torch.rand_like(mel_lens.float()))
            mel_lens = mel_lens.long()
            mel_max_len = torch.max(mel_lens).item()

        mel_masks = self.get_mask_from_lengths(mel_lens, mel_max_len)

        return dict(
            features=contents,
            cond_masks=src_masks,
            x_masks=mel_masks,
            x_lens=mel_lens,
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
        phones2mel=None,
        energy=None,
    ):
        features = self.forward_features(
            speakers=speakers,
            contents=contents,
            contents_lens=contents_lens,
            contents_max_len=contents_max_len,
            mel=mel,
            mel_lens=mel_lens,
            mel_max_len=mel_max_len,
            pitches=pitches,
            pitch_shift=pitch_shift,
            phones2mel=phones2mel,
            energy=energy,
        )

        if self.training:
            output_dict = self.diffusion.train_step(
                features["features"],
                mel,
                x_masks=features["x_masks"],
                cond_masks=features["cond_masks"],
            )
        else:
            output_dict = {
                "loss": 0.0,
            }

        if "loss" in features:
            output_dict["loss"] = output_dict["loss"] + features["loss"]

        if "metrics" in features:
            metrics = output_dict.get("metrics", {})
            metrics.update(features["metrics"])
            output_dict["metrics"] = metrics

        # For validation
        output_dict["features"] = features["features"]
        output_dict["x_masks"] = features["x_masks"]
        output_dict["x_lens"] = features["x_lens"]
        output_dict["cond_masks"] = features["cond_masks"]

        return output_dict
