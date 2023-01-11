import torch.nn as nn

import utils.pitch_tools
from utils.tools import get_mask_from_lengths

from diff_svc.diffusions import GaussianDiffusion


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffSinger, self).__init__()
        self.model = args.model
        self.model_config = model_config

        # self.text_encoder = FastspeechEncoder(model_config)
        self.diffusion = GaussianDiffusion(
            preprocess_config, model_config, train_config
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

        self.features_projection = nn.Linear(
            1024, model_config["transformer"]["encoder_hidden"]  # Hubert Embedding Size
        )

    def forward_features(
        self,
        speakers,
        contents,
        src_lens,
        max_src_len,
        mel_lens=None,
        max_mel_len=None,
        pitches=None
    ):

        features = self.features_projection(contents)
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
            pitches=pitches
        )

        output_dict = self.diffusion(
            features["features"],
            mels,
            features["mel_masks"]
        )

        # For validation
        output_dict["features"] = features["features"]

        return output_dict
