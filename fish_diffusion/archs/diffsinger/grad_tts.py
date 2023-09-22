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

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)
        self.duration_predictor = ENCODERS.build(model_config.duration_predictor)

        if getattr(model_config, "speaker_encoder", None):
            self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)

        if getattr(model_config, "gradient_checkpointing", False):
            self.text_encoder.bert.encoder.gradient_checkpointing = True

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

        features = self.text_encoder.bert.embeddings.word_embeddings(contents)[
            :, 0, :, :
        ]

        if speakers.ndim in [2, 3] and torch.is_floating_point(speakers):
            speaker_embed = speakers
        elif hasattr(self, "speaker_encoder"):
            speaker_embed = self.speaker_encoder(speakers)
        else:
            speaker_embed = None

        if speaker_embed is not None and speaker_embed.ndim == 2:
            speaker_embed = speaker_embed[:, None, :]

        # Ignore speaker embedding for now
        if speaker_embed is not None:
            features += speaker_embed

        src_masks_float = (~src_masks).to(features.dtype)
        features = self.text_encoder(
            inputs_embeds=features,
            attention_mask=src_masks_float,
            output_hidden_states=True,
        )

        # Predict durations
        log_durations = self.duration_predictor(features[:, 0, :])[..., 0]
        duration_loss = F.smooth_l1_loss(log_durations, torch.log(mel_lens.float()))

        if self.training is False:
            mel_lens = torch.round(torch.exp(torch.clamp(log_durations, 1, 8))).long()
            mel_lens = torch.clamp(mel_lens, 10, 2048)
            mel_max_len = torch.max(mel_lens).item()

        mel_masks = self.get_mask_from_lengths(mel_lens, mel_max_len)

        return dict(
            features=features,
            x_masks=mel_masks,
            x_lens=mel_lens,
            cond_masks=src_masks,
            loss=duration_loss,
            metrics={
                "duration_loss": duration_loss,
            },
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
