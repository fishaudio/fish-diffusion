import torch
import torch.nn as nn

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.vocoders.nsf_hifigan.models import AttrDict, Generator
from fish_diffusion.modules.vocoders.refinegan.generator import RefineGANGenerator


class HiFiSinger(nn.Module):
    """HiFiSinger"""

    def __init__(self, model_config):
        super(HiFiSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)

        if "pitch_shift_encoder" in model_config:
            self.pitch_shift_encoder = ENCODERS.build(model_config.pitch_shift_encoder)

        if "energy_encoder" in model_config:
            self.energy_encoder = ENCODERS.build(model_config.energy_encoder)

        self.feature_fuser = nn.Sequential(
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.SiLU(),
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.SiLU(),
        )

        if "type" in model_config.encoder and model_config.encoder.type == "RefineGAN":
            self.encoder_type = "RefineGAN"
            model_config.encoder.pop("type")
            self.encoder = RefineGANGenerator(**model_config.encoder)
        else:
            self.encoder_type = "HiFiGAN"
            self.encoder = Generator(AttrDict(model_config.encoder))

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
        pitch_shift=None,
        phones2mel=None,
        energy=None,
    ):
        src_masks = (
            self.get_mask_from_lengths(contents_lens, contents_max_len)
            if contents_lens is not None
            else None
        )

        features = self.text_encoder(contents, src_masks)

        if phones2mel is not None:
            phones2mel = (
                phones2mel.unsqueeze(-1).repeat([1, 1, features.shape[-1]]).long()
            )
            features = torch.gather(features, 1, phones2mel) * (
                1 - src_masks[:, :, None].float()
            )

        if speakers.ndim in [2, 3] and torch.is_floating_point(speakers):
            speaker_embed = speakers
        else:
            speaker_embed = self.speaker_encoder(speakers)

        if speaker_embed.ndim == 2:
            speaker_embed = speaker_embed[:, None, :]

        features += speaker_embed

        if pitch_shift is not None and hasattr(self, "pitch_shift_encoder"):
            pitch_shift_embed = self.pitch_shift_encoder(pitch_shift)

            if pitch_shift_embed.ndim == 2:
                pitch_shift_embed = pitch_shift_embed[:, None, :]

            features += pitch_shift_embed

        if energy is not None and hasattr(self, "energy_encoder"):
            energy_embed = self.energy_encoder(energy)

            if energy_embed.ndim == 2:
                energy_embed = energy_embed[:, None, :]

            features += energy_embed

        features = self.feature_fuser(features)
        features *= 1 - src_masks[:, :, None].float()

        return dict(
            features=features,
            src_masks=src_masks,
        )

    def forward(
        self,
        speakers,
        contents,
        contents_lens,
        contents_max_len,
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
            pitch_shift=pitch_shift,
            phones2mel=phones2mel,
            energy=energy,
        )

        if self.encoder_type == "RefineGAN":
            return self.encoder(
                features["features"].transpose(1, 2), pitches.transpose(1, 2)
            )
        else:
            return self.encoder(features["features"].transpose(1, 2), pitches[:, :, 0])
