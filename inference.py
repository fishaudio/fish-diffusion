import argparse
import math
import os
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from fish_audio_preprocess.utils import loudness_norm, separate_audio, slice_audio
from loguru import logger
from mmengine import Config

from fish_diffusion.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.utils.audio import get_mel_from_audio
from fish_diffusion.utils.pitch import PITCH_EXTRACTORS
from fish_diffusion.utils.tensor import repeat_expand_2d
from train import FishDiffusion


def slice_audio(
    audio: np.ndarray,
    rate: int,
    max_duration: float = 30.0,
    top_db: int = 60,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Iterable[tuple[int, int]]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        max_duration: maximum duration of each slice
        top_db: top_db of librosa.effects.split
        frame_length: frame_length of librosa.effects.split
        hop_length: hop_length of librosa.effects.split

    Returns:
        Iterable of start/end frame
    """

    intervals = librosa.effects.split(
        audio.T, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    for start, end in intervals:
        if end - start <= rate * max_duration:
            # Too short, unlikely to be vocal
            if end - start <= rate * 0.1:
                continue

            yield start, end
            continue

        n_chunks = math.ceil((end - start) / (max_duration * rate))
        chunk_size = math.ceil((end - start) / n_chunks)

        for i in range(start, end, chunk_size):
            yield i, i + chunk_size


@torch.no_grad()
def inference(
    config,
    checkpoint,
    input_path,
    output_path,
    speaker_id=0,
    pitch_adjust=0,
    silence_threshold=60,
    max_slice_duration=30.0,
    extract_vocals=True,
    merge_non_vocals=True,
    vocals_loudness_gain=0.0,
    device="cuda",
):
    """Inference

    Args:
        config: config
        checkpoint: checkpoint path
        input_path: input path
        output_path: output path
        speaker_id: speaker id
        pitch_adjust: pitch adjust
        silence_threshold: silence threshold of librosa.effects.split
        max_slice_duration: maximum duration of each slice
        extract_vocals: extract vocals
        merge_non_vocals: merge non-vocals, only works when extract_vocals is True
        vocals_loudness_gain: loudness gain of vocals (dB)
        device: device
    """

    if os.path.isdir(checkpoint):
        # Find the latest checkpoint
        checkpoints = sorted(os.listdir(checkpoint))
        checkpoint = os.path.join(checkpoint, checkpoints[-1])

    audio, sr = librosa.load(input_path, sr=config.sampling_rate, mono=True)

    # Extract vocals

    if extract_vocals:
        logger.info("Extracting vocals...")
        model = separate_audio.init_model("htdemucs", device=device)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.samplerate)[None]

        # To two channels
        audio = np.concatenate([audio, audio], axis=0)
        audio = torch.from_numpy(audio).to(device)
        tracks = separate_audio.separate_audio(
            model, audio, shifts=1, num_workers=0, progress=True
        )
        audio = separate_audio.merge_tracks(tracks, filter=["vocals"]).cpu().numpy()
        non_vocals = (
            separate_audio.merge_tracks(tracks, filter=["drums", "bass", "other"])
            .cpu()
            .numpy()
        )

        audio = librosa.resample(audio[0], orig_sr=model.samplerate, target_sr=sr)
        non_vocals = librosa.resample(
            non_vocals[0], orig_sr=model.samplerate, target_sr=sr
        )

        # Normalize loudness
        non_vocals = loudness_norm.loudness_norm(non_vocals, sr)

    # Normalize loudness
    audio = loudness_norm.loudness_norm(audio, sr)

    # Slice into segments
    segments = list(
        slice_audio(
            audio, sr, max_duration=max_slice_duration, top_db=silence_threshold
        )
    )
    logger.info(f"Sliced into {len(segments)} segments")

    # Load models
    text_features_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.text_features_extractor
    ).to(device)

    model = FishDiffusion(config)
    state_dict = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in state_dict:  # Checkpoint is saved by pl
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)

    pitch_extractor = PITCH_EXTRACTORS.get(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"

    generated_audio = np.zeros_like(audio)
    audio_torch = torch.from_numpy(audio).to(device)[None]

    for idx, (start, end) in enumerate(segments):
        segment = audio_torch[:, start:end]
        logger.info(
            f"Processing segment {idx + 1}/{len(segments)}, duration: {segment.shape[-1] / sr:.2f}s"
        )

        # Extract mel
        mel = get_mel_from_audio(segment, sr)

        # Extract pitch (f0)
        pitch = pitch_extractor(segment, sr, pad_to=mel.shape[-1]).float()
        pitch *= 2 ** (pitch_adjust / 12)

        # Extract text features
        text_features = text_features_extractor(segment, sr)[0]
        text_features = repeat_expand_2d(text_features, mel.shape[-1]).T

        # Predict
        src_lens = torch.tensor([mel.shape[-1]]).to(device)

        features = model.model.forward_features(
            speakers=torch.tensor([speaker_id]).long().to(device),
            contents=text_features[None].to(device),
            src_lens=src_lens,
            max_src_len=max(src_lens),
            mel_lens=src_lens,
            max_mel_len=max(src_lens),
            pitches=pitch[None].to(device),
        )

        result = model.model.diffusion.inference(features["features"], progress=False)
        wav = model.vocoder.spec2wav(result[0].T, f0=pitch).cpu().numpy()
        max_wav_len = generated_audio.shape[-1] - start
        generated_audio[start : start + wav.shape[-1]] = wav[:max_wav_len]

    # Loudness normalization
    generated_audio = loudness_norm.loudness_norm(generated_audio, sr)

    # Loudness gain
    loudness_float = 10 ** (vocals_loudness_gain / 20)
    generated_audio = generated_audio * loudness_float

    # Merge non-vocals
    if extract_vocals and merge_non_vocals:
        generated_audio = (generated_audio + non_vocals) / 2

    sf.write(output_path, generated_audio, sr)
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
        "--pitch_adjust",
        type=int,
        default=0,
        help="Pitch adjustment in semitones",
    )

    parser.add_argument(
        "--extract_vocals",
        action="store_true",
        help="Extract vocals",
    )

    parser.add_argument(
        "--merge_non_vocals",
        action="store_true",
        help="Merge non-vocals",
    )

    parser.add_argument(
        "--vocals_loudness_gain",
        type=float,
        default=0,
        help="Loudness gain for vocals",
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
        pitch_adjust=args.pitch_adjust,
        extract_vocals=args.extract_vocals,
        merge_non_vocals=args.merge_non_vocals,
        vocals_loudness_gain=args.vocals_loudness_gain,
        device=device,
    )
