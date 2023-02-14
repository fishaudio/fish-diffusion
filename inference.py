import argparse
import json
import os
from functools import partial
from typing import Union

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from fish_audio_preprocess.utils import loudness_norm, separate_audio
from loguru import logger
from mmengine import Config

from fish_diffusion.feature_extractors import FEATURE_EXTRACTORS, PITCH_EXTRACTORS
from fish_diffusion.utils.audio import get_mel_from_audio, slice_audio
from fish_diffusion.utils.inference import load_checkpoint
from fish_diffusion.utils.tensor import repeat_expand


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
    sampler_interval=None,
    sampler_progress=False,
    device="cuda",
    gradio_progress=None,
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
        sampler_interval: sampler interval, lower value means higher quality
        sampler_progress: show sampler progress
        device: device
        gradio_progress: gradio progress callback
    """

    if sampler_interval is not None:
        config.model.diffusion.sampler_interval = sampler_interval

    if os.path.isdir(checkpoint):
        # Find the latest checkpoint
        checkpoints = sorted(os.listdir(checkpoint))
        logger.info(f"Found {len(checkpoints)} checkpoints, using {checkpoints[-1]}")
        checkpoint = os.path.join(checkpoint, checkpoints[-1])

    audio, sr = librosa.load(input_path, sr=config.sampling_rate, mono=True)

    # Extract vocals

    if extract_vocals:
        logger.info("Extracting vocals...")

        if gradio_progress is not None:
            gradio_progress(0, "Extracting vocals...")

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
    text_features_extractor.eval()

    model = load_checkpoint(config, checkpoint, device=device)

    pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"

    generated_audio = np.zeros_like(audio)
    audio_torch = torch.from_numpy(audio).to(device)[None]

    for idx, (start, end) in enumerate(segments):
        if gradio_progress is not None:
            gradio_progress(idx / len(segments), "Generating audio...")

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
        text_features = repeat_expand(text_features, mel.shape[-1]).T

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

        result = model.model.diffusion.inference(
            features["features"], progress=sampler_progress
        )
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

    logger.info("Done")

    if output_path is not None:
        sf.write(output_path, generated_audio, sr)

    return generated_audio, sr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Run in gradio mode",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input audio file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to the output audio file",
    )

    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="Speaker id",
    )

    parser.add_argument(
        "--speaker_mapping",
        type=str,
        default=None,
        help="Speaker mapping file (gradio mode only)",
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


def run_inference(
    config_path: str,
    model_path: str,
    input_path: str,
    speaker: Union[int, str],
    pitch_adjust: int,
    sampler_interval: int,
    extract_vocals: bool,
    device: str,
    progress=gr.Progress(),
    speaker_mapping: dict = None,
):
    if speaker_mapping is not None and isinstance(speaker, str):
        speaker = speaker_mapping[speaker]

    audio, sr = inference(
        Config.fromfile(config_path),
        model_path,
        input_path=input_path,
        output_path=None,
        speaker_id=speaker,
        pitch_adjust=pitch_adjust,
        sampler_interval=round(sampler_interval),
        extract_vocals=extract_vocals,
        merge_non_vocals=False,
        device=device,
        gradio_progress=progress,
    )

    return (sr, audio)


def launch_gradio(args):
    with gr.Blocks(title="Fish Diffusion") as app:
        gr.Markdown("# Fish Diffusion SVC Inference")

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                    value=args.input,
                )
                output_audio = gr.Audio(label="Output Audio")

            with gr.Column():
                if args.speaker_mapping is not None:
                    speaker_mapping = json.load(open(args.speaker_mapping))

                    speaker = gr.Dropdown(
                        label="Speaker Name (Used for Multi-Speaker Models)",
                        choices=list(speaker_mapping.keys()),
                        value=list(speaker_mapping.keys())[0],
                    )
                else:
                    speaker_mapping = None
                    speaker = gr.Number(
                        label="Speaker ID (Used for Multi-Speaker Models)",
                        value=args.speaker_id,
                    )

                pitch_adjust = gr.Number(
                    label="Pitch Adjust (Semitones)", value=args.pitch_adjust
                )
                sampler_interval = gr.Slider(
                    label="Sampler Interval (⬆️ Faster Generation, ⬇️ Better Quality)",
                    value=args.sampler_interval or 10,
                    minimum=1,
                    maximum=100,
                )
                extract_vocals = gr.Checkbox(
                    label="Extract Vocals (For low quality audio)",
                    value=args.extract_vocals,
                )
                device = gr.Radio(
                    label="Device", choices=["cuda", "cpu"], value=args.device or "cuda"
                )

                run_btn = gr.Button(label="Run")

            run_btn.click(
                partial(
                    run_inference,
                    args.config,
                    args.checkpoint,
                    speaker_mapping=speaker_mapping,
                ),
                [
                    input_audio,
                    speaker,
                    pitch_adjust,
                    sampler_interval,
                    extract_vocals,
                    device,
                ],
                output_audio,
            )

    app.queue(concurrency_count=2).launch()


if __name__ == "__main__":
    args = parse_args()

    assert args.gradio or (
        args.input is not None and args.output is not None
    ), "Either --gradio or --input and --output should be specified"

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.gradio:
        args.device = device
        launch_gradio(args)

    else:

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
            sampler_interval=args.sampler_interval,
            sampler_progress=args.sampler_progress,
            device=device,
        )
