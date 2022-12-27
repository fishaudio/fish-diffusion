import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.pitch_tools
from .blocks import LinearNorm
from .modules import FastspeechEncoder, FastspeechDecoder, VarianceAdaptor
from .diffusion import GaussianDiffusion, GaussianDiffusionShallow
from utils.tools import get_mask_from_lengths


class DiffSinger(nn.Module):
    """ DiffSinger """

    def __init__(self, args, preprocess_config, model_config, train_config):
        super(DiffSinger, self).__init__()
        self.model = args.model
        self.model_config = model_config

        self.text_encoder = FastspeechEncoder(model_config)
        self.diffusion = None
        if self.model == "naive":
            self.diffusion = GaussianDiffusion(preprocess_config, model_config, train_config)
        elif self.model in ["aux", "shallow"]:
            self.decoder = FastspeechDecoder(model_config)
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
            self.diffusion = GaussianDiffusionShallow(preprocess_config, model_config, train_config)
        else:
            raise NotImplementedError

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
    def forward(
        self,
        speakers,
        contents,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        pitches=None
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        spk_emb =  self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        output = self.text_encoder(contents, src_masks,spk_emb)

        output += self.pitch_emb(utils.pitch_tools.f0_to_coarse(pitches))

        if self.model == "naive":
            (
                output,
                epsilon_predictions,
                noise_loss,
                diffusion_step,
            ) = self.diffusion(
                mels,
                output,
                mel_masks,
            )
        elif self.model in ["aux", "shallow"]:
            epsilon_predictions = noise_loss = diffusion_step = None
            cond = output.clone()
            output = self.decoder(output, mel_masks)
            output = self.mel_linear(output)
            self.diffusion.aux_mel = output.clone()
            if self.model == "shallow":
                (
                    output,
                    epsilon_predictions,
                    noise_loss,
                    diffusion_step,
                ) = self.diffusion(
                    mels,
                    cond,
                    mel_masks,
                )
        else:
            raise NotImplementedError

        return (
            output,
            epsilon_predictions,
            noise_loss,
            diffusion_step,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )