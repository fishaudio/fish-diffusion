import argparse
import os
import random
import subprocess as sp
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from random import Random
from typing import Optional

import librosa
import numpy as np
import torch
import torchcrepe
from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files
from loguru import logger
from mmengine import Config

from fish_diffusion.modules.energy_extractors import ENERGY_EXTRACTORS
from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.modules.feature_extractors.base import BaseFeatureExtractor
from fish_diffusion.modules.pitch_extractors import PITCH_EXTRACTORS
from fish_diffusion.modules.pitch_extractors.builder import BasePitchExtractor
from fish_diffusion.modules.vocoders import VOCODERS
from fish_diffusion.modules.vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN
from fish_diffusion.utils.tensor import repeat_expand

model_caches = None


def init(
    config,
) -> tuple[
    Optional[BaseFeatureExtractor],
    Optional[BasePitchExtractor],
    Optional[NsfHifiGAN],
    torch.device,
]:
    global model_caches
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger.info(f"{curr_worker} Uses device {device}")

    text_features_extractor = None
    if getattr(config.preprocessing, "text_features_extractor", None):
        text_features_extractor = FEATURE_EXTRACTORS.build(
            config.preprocessing.text_features_extractor
        )
        text_features_extractor.to(device)
        text_features_extractor.eval()

    pitch_extractor = None
    if getattr(config.preprocessing, "pitch_extractor", None):
        if config.preprocessing.pitch_extractor.type == "CrepePitchExtractor":
            torchcrepe.load.model(device, "full")

        pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
        pitch_extractor.to(device)
        pitch_extractor.eval()

    energy_extractor = None
    if getattr(config.preprocessing, "energy_extractor", None):
        energy_extractor = ENERGY_EXTRACTORS.build(
            config.preprocessing.energy_extractor
        )
        energy_extractor.to(device)
        energy_extractor.eval()

    vocoder = None
    if getattr(config.model, "vocoder", None):
        vocoder = VOCODERS.build(config.model.vocoder)
        vocoder.to(device)
        vocoder.eval()

    model_caches = (
        text_features_extractor,
        pitch_extractor,
        energy_extractor,
        vocoder,
        device,
    )


def process(
    config,
    audio_path: Path,
    idx: int = 0,
    key_shift: float = 0,
    time_stretch: float = 1.0,
    loudness: Optional[float] = None,
):
    if model_caches is None:
        init(config)
    (
        text_features_extractor,
        pitch_extractor,
        energy_extractor,
        vocoder,
        device,
    ) = model_caches

    save_path = audio_path.with_suffix(f".{idx}.data.npy")
    if save_path.exists():
        return

    sample = {
        "path": str(audio_path),
    }

    audio, sr = librosa.load(str(audio_path), sr=config.sampling_rate, mono=True)

    # Change loudness
    max_loudness = np.max(np.abs(audio))

    if loudness is not None:
        audio = audio * (loudness / (max_loudness + 1e-5))
    elif max_loudness > 1.0:
        audio = audio / (max_loudness + 1e-5)

    # If time_stretch is > 1, the audio length will be shorter (speed up)
    if time_stretch != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=time_stretch)

    sample["audio"] = audio
    sample["sampling_rate"] = sr
    sample["time_stretch"] = time_stretch

    # Move audio to appropriate device
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Obtain mel spectrogram
    if vocoder is not None:
        mel = vocoder.wav2spec(audio, sr, key_shift=key_shift)
        mel_length = mel.shape[-1]
        sample["mel"] = mel.cpu().numpy()
    else:
        # Calculate mel length from audio length
        hop_length = getattr(config, "hop_length", 512)
        mel_length = int(audio.shape[-1] / hop_length) + 1

    # Extract text features
    if text_features_extractor is not None:
        if config.model.type == "DiffSinger":
            contents, phones2mel = text_features_extractor(audio_path, mel_length)
            sample["phones2mel"] = phones2mel.cpu().numpy()
        if config.model.type == "GradTTS":
            contents = text_features_extractor(audio_path)
        else:
            contents = text_features_extractor(audio, sr)[0]
            contents = repeat_expand(contents, mel_length)

        sample["contents"] = contents.cpu().numpy()

    # Extract pitches
    if pitch_extractor is not None:
        pitches = pitch_extractor(audio, sr, pad_to=mel_length)
        pitches *= 2 ** (key_shift / 12)

        sample["pitches"] = pitches.cpu().numpy()
        sample["key_shift"] = key_shift  # Pitch is also gender params

    # Extract power
    if energy_extractor is not None:
        power = energy_extractor(audio, sr, pad_to=mel_length)
        sample["energy"] = power.cpu().numpy()

    # Save
    np.save(save_path, sample)


def safe_process(args, config, audio_path: Path):
    try:
        # Baseline
        process(config, audio_path)

        if args.no_augmentation or "augmentations" not in config.preprocessing:
            return 1

        # Augmentation
        augmentations = deepcopy(config.preprocessing.augmentations)
        aug_count = 0
        for augmentation in augmentations:
            probability = augmentation.probability

            while probability > 0:
                if random.random() > probability:
                    break

                probability -= 1
                aug_count += 1

                if augmentation.type == "FixedPitchShifting":
                    key_shift = random.choice(augmentation.key_shifts)
                    process(config, audio_path, idx=aug_count, key_shift=key_shift)
                elif augmentation.type == "RandomPitchShifting":
                    assert len(augmentation.key_shifts) == 2
                    key_shift = random.uniform(*augmentation.key_shifts)
                    process(config, audio_path, idx=aug_count, key_shift=key_shift)
                elif augmentation.type == "RandomTimeStretching":
                    assert len(augmentation.factors) == 2
                    factor = random.uniform(*augmentation.factors)
                    process(config, audio_path, idx=aug_count, time_stretch=factor)
                elif augmentation.type == "RandomLoudness":
                    assert len(augmentation.loudnesses) == 2
                    loudness = random.uniform(*augmentation.loudnesses)
                    process(config, audio_path, idx=aug_count, loudness=loudness)

        return aug_count + 1
    except Exception as e:
        logger.error(f"{curr_worker} Error processing {audio_path}")

        if args.debug:
            logger.exception(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers, will launch a process pool if > 1",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # For multiprocessing
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    curr_worker = f"[Rank {args.rank}]" if args.world_size > 1 else "[Main]"

    if torch.cuda.is_available():
        logger.info(f"{curr_worker} Found {torch.cuda.device_count()} GPUs")
    else:
        logger.warning(f"{curr_worker} No GPU found, using CPU")

    # Only clean on main process
    if args.clean and args.rank == 0:
        logger.info(f"{curr_worker} Cleaning *.npy files...")

        files = list_files(args.path, {".npy"}, recursive=True, sort=True)
        for f in files:
            f.unlink()

        logger.info(f"{curr_worker} Done!")

    # Multi-processing
    if args.num_workers > 1:
        logger.info(f"{curr_worker} Launching {args.num_workers} workers")

        processes = []
        for idx in range(args.num_workers):
            new_args = [
                "python",
                __file__,
                "--config",
                args.config,
                "--path",
                args.path,
                "--rank",
                str(idx),
                "--world-size",
                str(args.num_workers),
            ]

            if args.no_augmentation:
                new_args.append("--no-augmentation")

            if args.debug:
                new_args.append("--debug")

            env = deepcopy(os.environ)

            # Respect CUDA_VISIBLE_DEVICES
            if "CUDA_VISIBLE_DEVICES" in env:
                devices = env["CUDA_VISIBLE_DEVICES"].split(",")
                env["CUDA_VISIBLE_DEVICES"] = devices[idx % len(devices)]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(idx % torch.cuda.device_count())

            processes.append(sp.Popen(new_args, env=env))
            logger.info(f"{curr_worker} Launched worker {idx}")

        for p in processes:
            p.wait()

            if p.returncode != 0:
                logger.error(
                    f"{curr_worker} Worker {idx} failed with code {p.returncode}, exiting..."
                )
                exit(p.returncode)

        logger.info(f"{curr_worker} All workers done!")
        exit(0)

    # Load config
    config = Config.fromfile(args.config)
    files = list_files(args.path, AUDIO_EXTENSIONS, recursive=True, sort=True)

    # Shuffle files will balance the workload of workers
    Random(42).shuffle(files)

    logger.info(f"{curr_worker} Found {len(files)} files, processing...")

    # Chunk files
    if args.world_size > 1:
        files = files[args.rank :: args.world_size]
        logger.info(f"{curr_worker} Processing subset of {len(files)} files")

    # Main process
    total_samples, failed = 0, 0
    log_time = 0
    start_time = time.time()

    for idx, audio_path in enumerate(files):
        i = safe_process(args, config, audio_path)
        if isinstance(i, int):
            total_samples += i
        else:
            failed += 1

        if (idx + 1) % 100 == 0 and time.time() - log_time > 10:
            eta = (time.time() - start_time) / (idx + 1) * (len(files) - idx - 1)

            logger.info(
                f"{curr_worker} "
                + f"Processed {idx + 1}/{len(files)} files, "
                + f"{total_samples} samples, {failed} failed, "
                + f"ETA: {timedelta(seconds=eta)}"
            )

            log_time = time.time()

    logger.info(
        f"{curr_worker} Done! "
        + f"Original samples: {len(files)}, "
        + f"Augmented samples: {total_samples} (x{total_samples / len(files):.2f}), "
        + f"Failed: {failed}"
    )
