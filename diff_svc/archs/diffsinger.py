import torch.nn as nn
from diff_svc.denoisers.wavenet import WaveNetDenoiser

import utils.pitch_tools
from utils.tools import get_mask_from_lengths

from diff_svc.diffusions import GaussianDiffusion
from diff_svc.diffusions.modules import FastspeechEncoder


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.model_config = model_config

        self.text_encoder = FastspeechEncoder(model_config)
        denoiser = WaveNetDenoiser(
            mel_channels=128,
            d_encoder=256,
            residual_channels=512,
            residual_layers=20,
            dropout=0.2,
        )

        self.diffusion = GaussianDiffusion(
            denoiser,
            mel_channels=128,
            keep_bins=128,
            noise_schedule="linear",
            timesteps=1000,
            max_beta=0.01,
            s=0.008,
            noise_loss="smoothed-l1",
        )

        self.speaker_emb = None

        if model_config["multi_speaker"]:
            n_speakers = model_config["n_speakers"]

            self.speaker_emb = nn.Embedding(
                n_speakers,
                model_config["transformer"]["encoder_hidden"],
            )

        self.pitch_emb = nn.Embedding(
            256,
            model_config["transformer"]["encoder_hidden"],
        )

        if model_config["transformer"]["feature_distillation"]:
            self.features_projection = nn.Sequential(
                nn.Linear(
                    model_config["transformer"]["input_featuers"],
                    model_config["transformer"]["feature_distillation_dim"],
                ),
                nn.Softmax(dim=2),
                nn.Linear(
                    model_config["transformer"]["feature_distillation_dim"],
                    model_config["transformer"]["encoder_hidden"],
                ),
            )
        else:
            self.features_projection = nn.Linear(
                model_config["transformer"]["input_featuers"],
                model_config["transformer"]["encoder_hidden"],  # Hubert Embedding Size
            )

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

        features = self.features_projection(contents)
        x = self.features_projection[0](contents)
        x = self.features_projection[1](x)
        # import torch
        # print(x.shape, torch.topk(x, 3, dim=2))
        # from matplotlib import pyplot as plt
        # # histogram

        # plt.hist(x.flatten().cpu().numpy(), bins=100)
        # plt.savefig("hist.png")
        # exit()
        features = features + self.speaker_emb(speakers).unsqueeze(1).expand(
            -1, max_src_len, -1
        )
        features = features + self.pitch_emb(utils.pitch_tools.f0_to_coarse(pitches))

        src_masks = (
            get_mask_from_lengths(src_lens, max_src_len)
            if src_lens is not None
            else None
        )

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
