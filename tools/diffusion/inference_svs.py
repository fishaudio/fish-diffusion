import argparse
import json
import math
import os

import numpy as np
import soundfile as sf
import torch
from fish_audio_preprocess.utils import loudness_norm
from loguru import logger
from mmengine import Config

from fish_diffusion.archs.diffsinger import DiffSingerLightning
from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.modules.pitch_extractors import PITCH_EXTRACTORS
from fish_diffusion.utils.tensor import repeat_expand


@torch.no_grad()
def inference(
    config,
    checkpoint,
    input_path,
    output_path,
    dictionary_path="dictionaries/opencpop-strict.txt",
    speaker_id=0,
    sampler_interval=None,
    sampler_progress=False,
    device="cuda",
):
    """Inference

    Args:
        config: config
        checkpoint: checkpoint path
        input_path: input path
        output_path: output path
        dictionary_path: dictionary path
        speaker_id: speaker id
        sampler_interval: sampler interval, lower value means higher quality
        sampler_progress: show sampler progress
        device: device
    """

    if sampler_interval is not None:
        config.model.diffusion.sampler_interval = sampler_interval

    if os.path.isdir(checkpoint):
        # Find the latest checkpoint
        checkpoints = sorted(os.listdir(checkpoint))
        logger.info(f"Found {len(checkpoints)} checkpoints, using {checkpoints[-1]}")
        checkpoint = os.path.join(checkpoint, checkpoints[-1])

    # Load models
    phoneme_features_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.phoneme_features_extractor
    ).to(device)
    phoneme_features_extractor.eval()

    model = DiffSingerLightning(config)
    state_dict = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in state_dict:  # Checkpoint is saved by pl
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"

    # Load dictionary
    phones_list = []
    for i in open(dictionary_path):
        _, phones = i.strip().split("\t")
        for j in phones.split():
            if j not in phones_list:
                phones_list.append(j)

    phones_list = ["<PAD>", "<EOS>", "<UNK>", "AP", "SP"] + sorted(phones_list)

    # Load ds file
    with open(input_path) as f:
        ds = json.load(f)

    generated_audio = np.zeros(
        math.ceil(
            (
                float(ds[-1]["offset"])
                + float(ds[-1]["f0_timestep"]) * len(ds[-1]["f0_seq"].split(" "))
            )
            * config.sampling_rate
        )
    )

    for idx, chunk in enumerate(ds):
        offset = float(chunk["offset"])

        phones = np.array([phones_list.index(i) for i in chunk["ph_seq"].split(" ")])
        durations = np.array([0] + [float(i) for i in chunk["ph_dur"].split(" ")])
        durations = np.cumsum(durations)

        f0_timestep = float(chunk["f0_timestep"])
        f0_seq = torch.FloatTensor([float(i) for i in chunk["f0_seq"].split(" ")])
        f0_seq *= 2 ** (6 / 12)

        total_duration = f0_timestep * len(f0_seq)

        logger.info(
            f"Processing segment {idx + 1}/{len(ds)}, duration: {total_duration:.2f}s"
        )

        n_mels = round(total_duration * config.sampling_rate / 512)
        f0_seq = repeat_expand(f0_seq, n_mels, mode="linear")
        f0_seq = f0_seq.to(device)

        # aligned is in 20ms
        aligned_phones = torch.zeros(int(total_duration * 50), dtype=torch.long)
        for i, phone in enumerate(phones):
            start = int(durations[i] / f0_timestep / 4)
            end = int(durations[i + 1] / f0_timestep / 4)
            aligned_phones[start:end] = phone

        # Extract text features
        phoneme_features = phoneme_features_extractor.forward(
            aligned_phones.to(device)
        )[0]

        phoneme_features = repeat_expand(phoneme_features, n_mels).T

        # Predict
        src_lens = torch.tensor([phoneme_features.shape[0]]).to(device)

        features = model.model.forward_features(
            speakers=torch.tensor([speaker_id]).long().to(device),
            contents=phoneme_features[None].to(device),
            contents_lens=src_lens,
            contents_max_len=max(src_lens),
            mel_lens=src_lens,
            mel_max_len=max(src_lens),
            pitches=f0_seq[None],
        )

        result = model.model.diffusion(features["features"], progress=sampler_progress)
        wav = model.vocoder.spec2wav(result[0].T, f0=f0_seq).cpu().numpy()
        start = round(offset * config.sampling_rate)
        max_wav_len = generated_audio.shape[-1] - start
        generated_audio[start : start + wav.shape[-1]] = wav[:max_wav_len]

    # Loudness normalization
    generated_audio = loudness_norm.loudness_norm(generated_audio, config.sampling_rate)

    sf.write(output_path, generated_audio, config.sampling_rate)
    logger.info("Done")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/svc_hubert_soft.py",
        help="Path to the config file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input audio file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output audio file",
    )

    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker id",
    )

    parser.add_argument(
        "--sampler_interval",
        type=int,
        default=None,
        required=False,
        help="Sampler interval, if not specified, will be taken from config",
    )

    parser.add_argument(
        "--sampler_progress",
        action="store_true",
        help="Show sampler progress",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=False,
        help="Device to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    inference(
        Config.fromfile(args.config),
        args.checkpoint,
        args.input,
        args.output,
        speaker_id=args.speaker_id,
        sampler_interval=args.sampler_interval,
        sampler_progress=args.sampler_progress,
        device=device,
    )
