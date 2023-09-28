import argparse
import json
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from fish_audio_preprocess.utils import loudness_norm, separate_audio
from loguru import logger
from mmengine import Config
from natsort import natsorted
from torch import nn

from fish_diffusion.archs.diffsinger.diffsinger import DiffSingerLightning
from fish_diffusion.modules.energy_extractors import ENERGY_EXTRACTORS
from fish_diffusion.modules.feature_extractors import FEATURE_EXTRACTORS
from fish_diffusion.modules.pitch_extractors import PITCH_EXTRACTORS
from fish_diffusion.utils.audio import separate_vocals, slice_audio
from fish_diffusion.utils.inference import load_checkpoint
from fish_diffusion.utils.tensor import repeat_expand

# from tqdm import tqdm
# path = Path("dataset/train/aria")
# all_nps = list(path.rglob("*.npy"))[:5000]

# all_data = []
# for file in tqdm(all_nps):
#     try:
#         x = np.load(file, allow_pickle=True).item()
#         all_data.append(x['contents'].T)
#     except:
#         pass

# # import faiss
# XX = np.concatenate(all_data, axis=0).astype(np.float32)
# quantizer = faiss.IndexFlatL2(1024)
# index = faiss.IndexIVFFlat(quantizer, 1024, 100)
# assert not index.is_trained
# index.train(XX)
# assert index.is_trained
# index.add(XX)
# print("Index built")


class SVCInference(nn.Module):
    def __init__(self, config, checkpoint, model_cls=DiffSingerLightning):
        super().__init__()

        self.config = config

        self.text_features_extractor = FEATURE_EXTRACTORS.build(
            config.preprocessing.text_features_extractor
        )

        if getattr(config.preprocessing, "pitch_extractor", None):
            self.pitch_extractor = PITCH_EXTRACTORS.build(
                config.preprocessing.pitch_extractor
            )

        if getattr(config.preprocessing, "energy_extractor", None):
            self.energy_extractor = ENERGY_EXTRACTORS.build(
                config.preprocessing.energy_extractor
            )

        if os.path.isdir(checkpoint):
            # Find the latest checkpoint
            checkpoints = natsorted(os.listdir(checkpoint))
            logger.info(
                f"Found {len(checkpoints)} checkpoints, using {checkpoints[-1]}"
            )
            checkpoint = os.path.join(checkpoint, checkpoints[-1])

        self.model = load_checkpoint(
            config, checkpoint, device="cpu", model_cls=model_cls
        )
        self.separate_model = None

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        sr: int,
        pitch_adjust: int = 0,
        speakers: torch.Tensor = 0,
        sampler_progress: bool = False,
        sampler_interval: Optional[int] = None,
        noise_predictor: Optional[str] = None,
        pitches: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        if skip_steps > 0:
            original_mel = self.model.vocoder.wav2spec(audio, sr)[None]
            original_mel = original_mel.to(self.device)
            mel_len = original_mel.shape[-1]
        else:
            original_mel = None
            mel_len = audio.shape[-1] // 512

        # Extract and process pitch
        if hasattr(self, "pitch_extractor") and self.pitch_extractor is not None:
            if pitches is None:
                pitches = self.pitch_extractor(audio, sr, pad_to=mel_len).float()
            else:
                pitches = repeat_expand(pitches, mel_len)

            if (pitches == 0).all():
                return np.zeros((audio.shape[-1],))

            pitches *= 2 ** (pitch_adjust / 12)

        # Extract and process text features
        text_features = self.text_features_extractor(audio, sr)[0]
        text_features = repeat_expand(text_features, mel_len).T

        # Pitch shift should always be 0 for inference to avoid distortion
        pitch_shift = None
        if self.config.model.get("pitch_shift_encoder"):
            pitch_shift = torch.zeros((1, 1), device=self.device)

        energy = None
        if self.config.model.get("energy_encoder"):
            energy = self.energy_extractor(audio, sr, pad_to=mel_len)
            energy = energy[None, :, None]  # (1, mel_len, 1)

        # Predict
        contents_lens = torch.tensor([mel_len]).to(self.device)
        model = (
            self.model.ema_model
            if hasattr(self.model, "ema_model")
            else self.model.model
        )

        features = model.forward_features(
            speakers=speakers.to(self.device),
            contents=text_features[None].to(self.device),
            contents_lens=contents_lens,
            contents_max_len=max(contents_lens),
            mel_lens=contents_lens,
            mel_max_len=max(contents_lens),
            pitches=pitches[None].to(self.device) if pitches is not None else None,
            pitch_shift=pitch_shift,
            energy=energy,
        )

        result = model.diffusion(
            features["features"],
            progress=sampler_progress,
            sampler_interval=sampler_interval,
            noise_predictor=noise_predictor,
            skip_steps=skip_steps,
            original_mel=original_mel,
        )
        wav = self.model.vocoder.spec2wav(result[0].T, f0=pitches).cpu().numpy()

        return wav

    def _parse_speaker(self, speaker, recursive=True):
        to_long_tensor = lambda x: torch.tensor([x], dtype=torch.long)

        # Speaker id
        if isinstance(speaker, int):
            return to_long_tensor(speaker)

        # Speaker name
        if (
            hasattr(self.config, "speaker_mapping")
            and speaker in self.config.speaker_mapping
        ):
            return to_long_tensor(self.config.speaker_mapping[speaker])

        # Speaker id
        if speaker.isdigit():
            return to_long_tensor(int(speaker))

        if recursive is False:
            logger.error(f"Invalid speaker: {speaker}")
            exit()

        # Speaker mix
        speaker = speaker.split(",")
        speaker_mix = []

        for s in speaker:
            s = s.split(":")
            speaker_id = self._parse_speaker(s[0], recursive=False)

            if len(s) == 1:
                speaker_mix.append((speaker_id, 1.0))
            else:
                speaker_mix.append((speaker_id, float(s[1])))

        # Normalize speaker mix weights to 1
        summation = sum([s[1] for s in speaker_mix])
        speaker_mix = [(s[0], s[1] / summation) for s in speaker_mix]

        logger.info(
            f"Speaker mix: {speaker} -> {[f'{s[0].item()}:{s[1]}' for s in speaker_mix]}"
        )

        if hasattr(self.model, "model"):  # DiffSinger
            weight = self.model.model
        elif hasattr(self.model, "generator"):  # HiFiSinger
            weight = self.model.generator
        else:
            logger.error("Model does not have generator or model attribute")
            exit()

        weight = weight.speaker_encoder.embedding.weight
        mixed_weight = torch.zeros_like(weight[0])[None]
        for s in speaker_mix:
            mixed_weight += weight[s[0]] * s[1]

        return mixed_weight.float()

    @torch.no_grad()
    def inference(
        self,
        input_path,
        output_path,
        speaker=0,
        pitch_adjust=0,
        silence_threshold=60,
        max_slice_duration=30.0,
        extract_vocals=True,
        sampler_progress=False,
        sampler_interval=None,
        noise_predictor=None,
        gradio_progress=None,
        min_silence_duration=0,
        pitches_path=None,
        skip_steps=0,
    ):
        """Inference

        Args:
            input_path: input path
            output_path: output path
            speaker: speaker id or speaker name or speaker mix (a:0.5,b:0.5)
            pitch_adjust: pitch adjust
            silence_threshold: silence threshold of librosa.effects.split
            max_slice_duration: maximum duration of each slice
            extract_vocals: extract vocals
            sampler_progress: show sampler progress
            sampler_interval: sampler interval
            noise_predictor: noise predictor, can be naive, unipc, plms
            gradio_progress: gradio progress callback
            min_silence_duration: minimum silence duration
            pitches_path: disable pitch extraction and use the pitch from the given path
            skip_steps: skip steps
        """

        if isinstance(input_path, str) and os.path.isdir(input_path):
            if pitches_path is not None:
                logger.error("Pitch path is not supported for batch inference")
                return

            # Batch inference
            if output_path is None:
                logger.error("Output path is required for batch inference")
                return

            if os.path.exists(output_path) and not os.path.isdir(output_path):
                logger.error(
                    f"Output path {output_path} already exists, and it's not a directory"
                )
                return

            for file in os.listdir(input_path):
                self.inference(
                    input_path=os.path.join(input_path, file),
                    output_path=os.path.join(output_path, file),
                    speaker=speaker,
                    pitch_adjust=pitch_adjust,
                    silence_threshold=silence_threshold,
                    max_slice_duration=max_slice_duration,
                    extract_vocals=extract_vocals,
                    sampler_interval=sampler_interval,
                    sampler_progress=sampler_progress,
                    noise_predictor=noise_predictor,
                    gradio_progress=gradio_progress,
                    min_silence_duration=min_silence_duration,
                )

            return

        # Process speaker
        speakers = self._parse_speaker(speaker)
        if speakers is None:
            return

        # Load audio
        audio, sr = librosa.load(input_path, sr=self.config.sampling_rate, mono=True)

        logger.info(f"Loaded {input_path} with sr={sr}")

        # Extract vocals

        if extract_vocals:
            logger.info("Extracting vocals...")

            if gradio_progress is not None:
                gradio_progress(0, "Extracting vocals...")
            if self.separate_model is None:
                self.separate_model = separate_audio.init_model(
                    "htdemucs", device=self.device
                )
            audio, _ = separate_vocals(audio, sr, self.device, self.separate_model)

        # Normalize loudness
        audio = loudness_norm.loudness_norm(audio, sr)

        # Restore pitches if *.pitch.npy exists
        pitches = None

        if pitches_path is not None:
            logger.info(f"Restoring pitches from {pitches_path}")

            # If pitches_path is a json file, load it as a list of pitches
            if Path(pitches_path).suffix == ".json":
                with open(pitches_path, "r") as f:
                    pitches = json.load(f)
                    pitches = torch.FloatTensor(pitches).to(self.device)
            else:
                pitches = (
                    torch.from_numpy(np.load(pitches_path)).to(self.device).float()
                )

        # Slice into segments
        segments = list(
            slice_audio(
                audio,
                sr,
                max_duration=max_slice_duration,
                top_db=silence_threshold,
                min_silence_duration=min_silence_duration,
            )
        )
        logger.info(f"Sliced into {len(segments)} segments")

        generated_audio = np.zeros_like(audio)
        audio_torch = torch.from_numpy(audio).to(self.device)[None]

        for idx, (start, end) in enumerate(segments):
            if gradio_progress is not None:
                gradio_progress(idx / len(segments), "Generating audio...")

            segment = audio_torch[:, start:end]
            logger.info(
                f"Processing segment {idx + 1}/{len(segments)}, duration: {segment.shape[-1] / sr:.2f}s"
            )

            pitches_segment = None
            if pitches is not None:
                pitches_segment = pitches[start // 512 : end // 512]
                pitches_segment[torch.isnan(pitches_segment)] = 0

            wav = self(
                segment,
                sr,
                pitch_adjust=pitch_adjust,
                speakers=speakers,
                sampler_progress=sampler_progress,
                sampler_interval=sampler_interval,
                noise_predictor=noise_predictor,
                pitches=pitches_segment,
                skip_steps=skip_steps,
            )
            max_wav_len = generated_audio.shape[-1] - start
            generated_audio[start : start + wav.shape[-1]] = wav[:max_wav_len]

        # Loudness normalization (disabled)
        # generated_audio = loudness_norm.loudness_norm(generated_audio, sr)

        logger.info("Done")

        if output_path is not None:
            if os.path.exists(os.path.dirname(output_path)) is False:
                os.makedirs(os.path.dirname(output_path))

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
        "--gradio_share",
        action="store_true",
        help="Share gradio app",
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
        "--speaker",
        type=str,
        default="0",
        help="Speaker id or speaker name (if speaker_mapping is specified) or speaker mix (a:0.5,b:0.5)",
    )

    parser.add_argument(
        "--speaker_mapping",
        type=str,
        default=None,
        help="Speaker mapping file (if not specified, will be taken from config)",
    )

    parser.add_argument(
        "--pitch_adjust",
        type=int,
        default=0,
        help="Pitch adjustment in semitones",
    )

    parser.add_argument(
        "--pitches_path",
        type=str,
        default=None,
        help="Path to the pitch file",
    )

    parser.add_argument(
        "--extract_vocals",
        action="store_true",
        help="Extract vocals",
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
        "--noise_predictor",
        type=str,
        default=None,
        required=False,
        help="Noise predictor, can be naive, unipc, plms",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=False,
        help="Device to use",
    )

    # Slicer arguments
    parser.add_argument(
        "--silence_threshold",
        type=int,
        default=60,
        help="Silence threshold in dB",
    )

    parser.add_argument(
        "--max_slice_duration",
        type=int,
        default=30,
        help="Max slice duration in seconds",
    )

    parser.add_argument(
        "--min_silence_duration",
        type=int,
        default=0,
        help="Min silence duration in seconds",
    )

    # Pitch extractor
    parser.add_argument(
        "--pitch_extractor",
        type=str,
        default=None,
        help="Pitch extractor",
    )

    # Shallow diffusion
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=0,
        help="Skip steps and use original audio as input",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.gradio or (
        args.input is not None and args.output is not None
    ), "Either --gradio or --input and --output should be specified"

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = Config.fromfile(args.config)

    if args.speaker_mapping is not None:
        config.speaker_mapping = json.load(open(args.speaker_mapping))

    if args.pitch_extractor is not None:
        config.preprocessing.pitch_extractor.type = args.pitch_extractor

    model = SVCInference(config, args.checkpoint)
    model = model.to(device)

    if args.gradio:
        from tools.diffusion.gradio_ui import launch_gradio

        launch_gradio(
            config,
            model.inference,
            speaker=args.speaker,
            pitch_adjust=args.pitch_adjust,
            sampler_interval=args.sampler_interval,
            extract_vocals=args.extract_vocals,
            share=args.gradio_share,
        )

    else:
        model.inference(
            input_path=args.input,
            output_path=args.output,
            speaker=args.speaker,
            pitch_adjust=args.pitch_adjust,
            pitches_path=args.pitches_path,
            extract_vocals=args.extract_vocals,
            sampler_progress=args.sampler_progress,
            sampler_interval=args.sampler_interval,
            noise_predictor=args.noise_predictor,
            silence_threshold=args.silence_threshold,
            max_slice_duration=args.max_slice_duration,
            min_silence_duration=args.min_silence_duration,
            skip_steps=args.skip_steps,
        )
