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
    model = DiffSingerLightning(config)
    state_dict = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in state_dict:  # Checkpoint is saved by pl
        state_dict = state_dict["state_dict"]

    x = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"

    # Load dictionary
    phones_list = config.phonemes

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

        # Merge slurs
        phones, durations = [], []
        for phone, duration, is_slur in zip(
            chunk["ph_seq"].split(" "),
            chunk["ph_dur"].split(" "),
            chunk["is_slur_seq"].split(" "),
        ):
            if is_slur == "1":
                durations[-1] = durations[-1] + float(duration)
            else:
                phones.append(phones_list.index(phone))
                durations.append(float(duration))

        phones = np.array(phones)
        durations = np.array(durations)

        # phones = np.array([phones_list.index(i) for i in chunk["ph_seq"].split(" ")])
        # durations = np.array([float(i) for i in chunk["ph_dur"].split(" ")])

        f0_timestep = float(chunk["f0_timestep"])
        f0_seq = torch.FloatTensor([float(i) for i in chunk["f0_seq"].split(" ")])
        # f0_seq *= 2 ** (0 / 12)

        total_duration = f0_timestep * len(f0_seq)

        logger.info(
            f"Processing segment {idx + 1}/{len(ds)}, duration: {total_duration:.2f}s"
        )

        n_mels = round(total_duration * config.sampling_rate / 512)

        t_max = (len(f0_seq) - 1) * f0_timestep
        dt = 512 / config.sampling_rate
        f0_seq = np.interp(
            np.arange(0, t_max, dt), f0_timestep * np.arange(len(f0_seq)), f0_seq
        )
        f0_seq = torch.from_numpy(f0_seq).type(torch.float32)
        f0_seq = repeat_expand(f0_seq, n_mels, mode="linear")

        f0_seq = f0_seq.to(device)

        # aligned is in 20ms
        cumsum_durations = np.cumsum(durations)
        alignment_factor = n_mels / cumsum_durations[-1]
        num_classes = len(phones_list)

        features = torch.zeros((n_mels, num_classes * 2 + 2), dtype=torch.float32)

        for i, (phone, duration, sum_duration) in enumerate(
            zip(phones, durations, cumsum_durations)
        ):
            current_idx = int(sum_duration * alignment_factor)
            previous_idx = (
                int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
            )
            _temp = torch.zeros(num_classes * 2 + 1, dtype=torch.float32)

            if i > 0:
                # Previous phoneme
                _temp[phones[i - 1]] = 1

            _temp[num_classes + phone] = 1
            _temp[-1] = duration

            features[previous_idx:current_idx, : num_classes * 2 + 1] = _temp

            # End of phoneme
            features[previous_idx, -1] = 1

        phoneme_features = features.to(device)
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

        # Save wav
        # sf.write("test.wav", wav, config.sampling_rate)
        # exit()

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
