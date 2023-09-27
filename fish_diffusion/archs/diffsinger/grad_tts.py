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
        self.bert_query = nn.Parameter(torch.randn(1, 1, self.text_encoder.output_size))

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
        if self.training is False:
            # Random 20% size change
            mel_lens = torch.round(
                mel_lens * (0.8 + 0.4 * torch.rand(1, device=mel_lens.device))
            ).long()
            mel_max_len = torch.max(mel_lens).item()

        # Build text features
        text_features = self.text_encoder.bert.embeddings.word_embeddings(contents)[
            :, 0, :, :
        ]
        text_masks = self.get_mask_from_lengths(contents_lens, contents_max_len)

        # Build Query features
        query_lengths = (mel_lens / 10).ceil().long()  # 512 * 10 / 44100 = 0.116 sec
        max_query_length = torch.max(query_lengths).item()
        mel_queries = self.bert_query.expand(mel.shape[0], max_query_length, -1)
        mel_masks = self.get_mask_from_lengths(query_lengths, max_query_length)

        # Concatenate text and query features
        # This will waste memory, is there any better way?
        features = torch.cat([text_features, mel_queries], dim=1)
        src_masks = torch.cat([text_masks, mel_masks], dim=1)

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

        # Let's extract query features
        features = features[:, -max_query_length:, :]

        # Repeat to match mel length
        features = features.repeat_interleave(10, dim=1)

        # Truncate to match mel length
        features = features[:, :mel_max_len, :]
        mel_masks = self.get_mask_from_lengths(mel_lens, mel_max_len)

        return dict(
            features=features,
            x_masks=mel_masks,
            x_lens=mel_lens,
            cond_masks=mel_masks,
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
