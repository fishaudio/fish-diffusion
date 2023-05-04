import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
import torchcrepe
from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files
from loguru import logger

# from mmengine import Config
from tqdm import tqdm

# from fish_diffusion.modules.energy_extractors import ENERGY_EXTRACTORS
# from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.modules.feature_extractors.base import BaseFeatureExtractor

# from fish_diffusion.modules.pitch_extractors import PITCH_EXTRACTORS
from fish_diffusion.modules.pitch_extractors.builder import BasePitchExtractor

# from fish_diffusion.modules.vocoders import VOCODERS
from fish_diffusion.modules.vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN
from fish_diffusion.utils.tensor import repeat_expand

from box import Box
from omegaconf import OmegaConf, DictConfig
import hydra
from tools.diffusion import resolvers
from hydra.utils import instantiate

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

    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0

    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")

    logger.info(f"Rank {rank} uses device {device}")

    text_features_extractor = None
    if hasattr(config.preprocessing, "text_features_extractor"):
        text_features_extractor = instantiate(
            config.preprocessing.text_features_extractor
        )
        text_features_extractor.to(device)
        text_features_extractor.eval()

    pitch_extractor = None
    if hasattr(config.preprocessing, "pitch_extractor"):
        if config.preprocessing.pitch_extractor.type == "CrepePitchExtractor":
            torchcrepe.load.model(device, "full")

        pitch_extractor = instantiate(config.preprocessing.pitch_extractor)

    energy_extractor = None
    if hasattr(config.preprocessing, "energy_extractor"):
        energy_extractor = instantiate(config.preprocessing.energy_extractor)
        energy_extractor.to(device)
        energy_extractor.eval()

    vocoder = None
    if hasattr(config.model, "vocoder"):
        vocoder = instantiate(config.model.vocoder)
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

        return aug_count + 1
    except Exception as e:
        logger.error(f"Error processing {audio_path}")
        logger.exception(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--no-augmentation", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    logger.info(f"Using {args.num_workers} workers")

    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPUs")
    else:
        logger.warning("No GPU found, using CPU")

    if args.clean:
        logger.info("Cleaning *.npy files...")

        files = list_files(args.path, {".npy"}, recursive=True, sort=True)
        for f in files:
            f.unlink()

        logger.info("Done!")

    config = Config.fromfile(args.config)
    # files = list_files(args.path, AUDIO_EXTENSIONS, recursive=True, sort=False)
    files = list(Path(args.path).glob("*/**/*.wav"))
    logger.info(f"Found {len(files)} files, processing...")

    # Shuffle files will balance the workload of workers
    random.shuffle(files)
    total_samples, failed = 0, 0

    if args.num_workers <= 1:
        for audio_path in tqdm(files):
            i = safe_process(args, config, audio_path)
            if isinstance(i, int):
                total_samples += i
            else:
                failed += 1
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
        ) as executor:
            params = [(args, config, audio_path) for audio_path in files]

            for i in tqdm(executor.map(safe_process, *zip(*params)), total=len(params)):
                if isinstance(i, int):
                    total_samples += i
                else:
                    failed += 1

    logger.info(f"Finished!")
    logger.info(f"Original samples: {len(files)}")
    logger.info(
        f"Augmented samples: {total_samples} (x{total_samples / len(files):.2f})"
    )
    logger.info(f"Failed: {failed}")
