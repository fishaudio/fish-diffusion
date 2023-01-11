import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data_utils import Dataset
from model import DiffSingerLoss
from utils.tools import log, synth_one_sample, to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        preprocess_config["path"]["val_filelist"],
        preprocess_config,
        train_config,
        sort=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    loss_sums = [
        {k: 0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses
    ]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            pitches = batch[8].clone()
            with torch.no_grad():
                # Forward
                output = model(*(batch[1:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k: v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}, ".format(
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            args,
            batch,
            pitches,
            output,
            vocoder,
            model_config,
            preprocess_config,
            model.module.diffusion,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/{}_reconstructed".format(tag),
            step=step,
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/{}_synthesized".format(tag),
            step=step,
        )

    return message
