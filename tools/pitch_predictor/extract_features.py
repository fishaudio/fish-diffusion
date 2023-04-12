import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
import torchcrepe
import torchvision
from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from mmengine import Config
from tqdm import tqdm

from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.modules.feature_extractors.base import BaseFeatureExtractor
from fish_diffusion.modules.pitch_extractors import PITCH_EXTRACTORS
from fish_diffusion.modules.pitch_extractors.builder import BasePitchExtractor
from fish_diffusion.modules.vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN
from fish_diffusion.utils.pitch import pitch_to_mel

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
        text_features_extractor = FEATURE_EXTRACTORS.build(
            config.preprocessing.text_features_extractor
        )
        text_features_extractor.to(device)
        text_features_extractor.eval()

    pitch_extractor = None
    if hasattr(config.preprocessing, "pitch_extractor"):
        if config.preprocessing.pitch_extractor.type == "CrepePitchExtractor":
            torchcrepe.load.model(device, "full")

        pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)

    model_caches = (
        text_features_extractor,
        pitch_extractor,
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

    (text_features_extractor, pitch_extractor, device) = model_caches

    save_path = audio_path.with_suffix(f".{idx}.data.npy")
    if save_path.exists():
        return

    sample = {
        "path": str(audio_path),
    }

    audio, sr = librosa.load(str(audio_path), sr=config.sampling_rate, mono=True)

    sample["audio"] = audio
    sample["sampling_rate"] = sr
    sample["time_stretch"] = 1.0
    sample["key_shift"] = 0

    # Move audio to appropriate device
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Calculate mel length from audio length
    mel_length = int(audio.shape[-1] / 512) + 1

    # Extract text features
    contents, phones2mel = text_features_extractor(audio_path, mel_length)

    sample["phones2mel"] = phones2mel.cpu().numpy()
    sample["contents"] = contents.cpu().numpy()

    f0 = text_features_extractor.notes_f0(audio_path).to(device)
    f0 = torch.gather(f0, 0, phones2mel.to(device))
    sample["pitches"] = f0.cpu().numpy()

    # Extract pitches
    predicted_pitches = pitch_extractor(audio, sr, pad_to=mel_length)
    mel_scale = pitch_to_mel(
        predicted_pitches, config.f0_mel_min, config.f0_mel_max, config.mel_bins
    )

    # Remove zeros
    mel_scale[predicted_pitches == 0] = 0
    # Apply gaussian blur (will perform better on CenterNet, but not sure for diffuision)
    mel_scale = torchvision.transforms.functional.gaussian_blur(mel_scale[None], 3)[0]

    # print(mel_scale.shape)
    sample["mel"] = mel_scale.cpu().numpy().T

    # from matplotlib import pyplot as plt
    # plt.imshow(mel_scale.cpu().numpy().T)
    # plt.savefig("mel_scale.png")
    # plt.plot(predicted_pitches.cpu().numpy())
    # plt.plot(f0.cpu().numpy())
    # plt.savefig("pitches.png")
    # exit()

    # Save
    np.save(save_path, sample)


def safe_process(args, config, audio_path: Path):
    try:
        # Baseline
        process(config, audio_path)
        return 1
    except Exception as e:
        logger.error(f"Error processing {audio_path}")
        logger.exception(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)

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
    files = list_files(args.path, {".wav"}, recursive=True, sort=False)
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
                exit()
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
    logger.info(f"Samples: {len(files)}")
    logger.info(f"Failed: {failed}")
