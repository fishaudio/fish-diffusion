import math

import torch
import torch.nn as nn

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.monotonic_align import maximum_path

from .diffusions import DIFFUSIONS


class GradTTS(nn.Module):
    """GradTTS"""

    def __init__(self, model_config):
        super(GradTTS, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)
        self.mel_encoder = ENCODERS.build(model_config.mel_encoder)
        self.duration_predictor = ENCODERS.build(model_config.duration_predictor)

        if getattr(model_config, "speaker_encoder", None):
            self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)

        self.current_mas_noise_scale = nn.Parameter(
            torch.tensor(1e-2, dtype=torch.float32), requires_grad=False
        )

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

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

    @staticmethod
    def convert_pad_shape(pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape

    @staticmethod
    def generate_path(duration, mask):
        device = duration.device

        b, t_x, t_y = mask.shape
        cum_duration = torch.cumsum(duration, 1)
        path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

        cum_duration_flat = cum_duration.view(b * t_x)
        path = GradTTS.sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        path = path.view(b, t_x, t_y)
        path = (
            path
            - torch.nn.functional.pad(
                path, GradTTS.convert_pad_shape([[0, 0], [1, 0], [0, 0]])
            )[:, :-1]
        )
        path = path * mask

        return path

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
        src_masks = (
            self.get_mask_from_lengths(contents_lens, contents_max_len)
            if contents_lens is not None
            else None
        )
        src_masks = (~src_masks).float()

        mel_masks = (
            self.get_mask_from_lengths(mel_lens, mel_max_len)
            if mel_lens is not None
            else None
        )

        features = self.text_encoder(contents, src_masks)

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

        stats_p = self.mel_encoder(features, src_masks).transpose(1, 2)
        mu_p, logs_p = torch.split(stats_p, stats_p.shape[1] // 2, dim=1)
        logw = self.duration_predictor(features.detach(), src_masks).transpose(1, 2)
        x_mask = src_masks.unsqueeze(1)

        if not self.training:
            # We are in validation mode
            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = int(y_lengths.max())

            # Using obtained durations `w` construct alignment map `attn`
            y_mask = (
                GradTTS.sequence_mask(y_lengths, y_max_length)
                .unsqueeze(1)
                .to(x_mask.dtype)
            )
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = GradTTS.generate_path(
                w_ceil.squeeze(1), attn_mask.squeeze(1)
            ).unsqueeze(1)

            # Align encoded text and get mu_y
            z_p = mu_p + torch.randn_like(mu_p) * torch.exp(logs_p) * 0.667
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), z_p.transpose(1, 2))
            mel_features = mu_y[:, :y_max_length, :]

            return dict(
                features=mel_features,
                mel_masks=mel_masks,
            )

        z_p = mel.transpose(1, 2)
        y_mask = (~mel_masks).float().unsqueeze(1)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (mu_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (mu_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            # Noise scale
            epsilon = (
                torch.std(neg_cent)
                * torch.randn_like(neg_cent)
                * self.current_mas_noise_scale
            )
            neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + attn.sum(2)) * x_mask
        dur_loss = torch.sum((logw - logw_) ** 2) / torch.sum(src_masks)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_p = torch.matmul(attn.squeeze(1), mu_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p_hat = mu_p + torch.randn_like(mu_p) * torch.exp(logs_p) * 0.667

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(
            0.5 * ((z_p - z_p_hat) ** 2 + math.log(2 * math.pi)) * y_mask
        )
        prior_loss = prior_loss / (torch.sum(y_mask) * z_p.shape[-2])

        return dict(
            features=z_p_hat.transpose(1, 2),
            mel_masks=mel_masks,
            loss=dur_loss + prior_loss,
            metrics={
                "dur_loss": dur_loss,
                "prior_loss": prior_loss,
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
                features["features"], mel, features["mel_masks"]
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

        if self.training:
            output_dict["metrics"]["noise_scale"] = float(self.current_mas_noise_scale)

            # Update MAS noise scale
            self.current_mas_noise_scale -= 2e-6
            if self.current_mas_noise_scale < 0.0:
                self.current_mas_noise_scale -= (
                    self.current_mas_noise_scale
                )  # clip to 0

        # For validation
        output_dict["features"] = features["features"]

        return output_dict
