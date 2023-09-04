import argparse
import json
from typing import Optional

import numpy as np
import torch
from mmengine import Config

from fish_diffusion.archs.hifisinger import HiFiSingerV1Lightning, HiFiSingerV2Lightning
from fish_diffusion.utils.tensor import repeat_expand
from tools.diffusion.inference import SVCInference


class HiFiSingerSVCInference(SVCInference):
    def __init__(self, config, checkpoint):
        if config.model.encoder.type.lower() == "RefineGAN".lower():
            model_cls = HiFiSingerV2Lightning
        elif config.model.encoder.type.lower() == "HiFiGAN".lower():
            model_cls = HiFiSingerV1Lightning
        else:
            raise NotImplementedError(
                f"Unknown encoder type: {config.model.encoder.type}"
            )

        super().__init__(config, checkpoint, model_cls=model_cls)

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
        skip_steps: int = 0,  # not used
    ):
        mel_len = audio.shape[-1] // getattr(self.config, "hop_length", 512)
        amplitude = audio.abs().max()

        # Extract and process pitch
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

        wav = self.model.generator(
            speakers=speakers.to(self.device),
            contents=text_features[None].to(self.device),
            contents_lens=contents_lens,
            contents_max_len=max(contents_lens),
            pitches=pitches[None, :, None].to(self.device),
            pitch_shift=pitch_shift,
            energy=energy,
        )

        wav_amplitude = wav.abs().max()
        wav *= amplitude / wav_amplitude

        return wav.cpu().numpy()[0]


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

    model = HiFiSingerSVCInference(config, args.checkpoint)
    model = model.to(device)

    model.inference(
        input_path=args.input,
        output_path=args.output,
        speaker=args.speaker,
        pitch_adjust=args.pitch_adjust,
        pitches_path=args.pitches_path,
        extract_vocals=args.extract_vocals,
        silence_threshold=args.silence_threshold,
        max_slice_duration=args.max_slice_duration,
        min_silence_duration=args.min_silence_duration,
    )
