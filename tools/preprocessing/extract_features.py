import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock, Value
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe
from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from mmengine import Config
from tqdm import tqdm

from fish_diffusion.feature_extractors import FEATURE_EXTRACTORS, PITCH_EXTRACTORS
from fish_diffusion.utils.tensor import repeat_expand
from fish_diffusion.vocoders import VOCODERS

text_features_extractor = None
vocoder = None
device = None


def init(worker_id: Value, config):
    global text_features_extractor, vocoder, device

    with worker_id.get_lock():
        current_id = worker_id.value
        worker_id.value += 1

    device = torch.device(
        f"cuda:{current_id % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )

    if config.preprocessing.text_features_extractor is not None:
        text_features_extractor = FEATURE_EXTRACTORS.build(
            config.preprocessing.text_features_extractor
        )
        text_features_extractor.to(device)
        text_features_extractor.eval()

    if config.preprocessing.pitch_extractor == "crepe":
        torchcrepe.load.model(device, "full")

    vocoder = VOCODERS.build(config.model.vocoder)


def process(config, audio_path: Path, override: bool = False):
    # Important for multiprocessing
    global text_features_extractor, vocoder, device

    audio, sr = librosa.load(str(audio_path), sr=config.sampling_rate, mono=True)
    # audio: (1, T)

    # Move audio to appropriate device
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Obtain mel spectrogram
    mel_path = audio_path.parent / f"{audio_path.name}.mel.npy"

    if mel_path.exists() is False or override:
        mel = vocoder.wav2spec(audio, sr)
        np.save(mel_path, mel.cpu().numpy())
    else:
        mel = np.load(mel_path)

    # Extract text features
    text_features_path = audio_path.parent / f"{audio_path.name}.text_features.npy"

    if (
        text_features_extractor is not None
        and text_features_path.exists() is False
        or override
    ):
        if config.model.type == "DiffSinger":
            text_features = text_features_extractor(audio_path, mel.shape[-1])
        else:
            text_features = text_features_extractor(audio, sr)[0]
            text_features = repeat_expand(text_features, mel.shape[-1])

        np.save(text_features_path, text_features.cpu().numpy())

    # Extract f0
    f0_path = audio_path.parent / f"{audio_path.name}.f0.npy"

    if f0_path.exists() is False or override:
        pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)

        assert (
            pitch_extractor is not None
        ), f"Unknown pitch extractor: {config.preprocessing.pitch_extractor}"

        f0 = pitch_extractor(audio, sr, pad_to=mel.shape[-1])
        np.save(f0_path, f0.cpu().numpy())


def safe_process(config, audio_path: Path, override: bool = False):
    try:
        process(config, audio_path, override)
    except Exception as e:
        logger.error(f"Error processing {audio_path}")
        logger.exception(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn")

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
    files = list_files(args.path, {".wav"}, recursive=True, sort=True)

    logger.info(f"Found {len(files)} files, processing...")

    worker_id = Value("i", 0)

    if args.num_workers <= 1:
        init(worker_id, config)

        for audio_path in tqdm(files):
            safe_process(config, audio_path, args.override)
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init,
            initargs=(worker_id, config),
        ) as executor:
            params = [(config, audio_path, args.override) for audio_path in files]

            for i in tqdm(executor.map(safe_process, *zip(*params)), total=len(params)):
                assert i is None, i

    logger.info("Done!")
