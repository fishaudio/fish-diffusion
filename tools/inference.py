import torch
from pathlib import Path
from box import Box

import click
import hydra
from loguru import logger
from omegaconf import OmegaConf


@click.command()
@click.option("--input", required=True, help="Path to input file")
@click.option("--output", required=True, help="Path to output file")
@click.option("--speaker", default=None, help="Speaker ID")
@click.option("--pitch-adjust", default=0.0, help="Pitch adjustment")
@click.option("--pitches-path", default=None, help="Path to pitch contour file")
@click.option("--extract-vocals", is_flag=True, help="Extract vocals")
@click.option("--sampler-progress", is_flag=True, help="Show sampler progress")
@click.option("--sampler-interval", default=0.1, help="Sampler interval")
@click.option("--silence-threshold", default=0.1, help="Silence threshold")
@click.option("--max-slice-duration", default=10.0, help="Maximum slice duration")
@click.option("--min-silence-duration", default=0.5, help="Minimum silence duration")
@click.option("--skip-steps", default=None, help="Steps to skip")
@click.option("--config", default="config.yaml", help="Config file")
@click.option("--checkpoint", default="best_model.pth", help="Checkpoint file")
@click.option("--help", is_flag=True, help="Show help")
def run_inference(
    input,
    output,
    speaker,
    pitch_adjust,
    pitches_path,
    extract_vocals,
    sampler_progress,
    sampler_interval,
    silence_threshold,
    max_slice_duration,
    min_silence_duration,
    skip_steps,
    config,
    checkpoint,
    help,
):
    if help:
        click.echo(click.get_current_context().get_help())
        return
    run_dir = Path(config).parent
    config = Path(config).stem
    logger.info(f"using {config} in {run_dir}")
    with hydra.initialize(config_path=f"../{run_dir}", job_name=run_dir):
        cfg = hydra.compose(config_name=config)
        model_type = cfg.model.type
        if model_type == "HiFiSingerSVC":
            from hifisinger.inference import HiFiSingerSVCInference as Inference
        elif model_type == "DiffSVC":
            from tools.diffusion.inference import SVCInference as Inference
        else:
            logger.error(f"Unknown model type {model_type}")
            raise ValueError(f"Unknown model type {model_type}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        config = OmegaConf.load(config)
        config = Box(config)

        model = Inference(config, checkpoint)
        model = model.to(device)
        model.inference(
            input_path=input,
            output_path=output,
            speaker=speaker,
            pitch_adjust=pitch_adjust,
            pitches_path=pitches_path,
            extract_vocals=extract_vocals,
            sampler_progress=sampler_progress,
            sampler_interval=sampler_interval,
            silence_threshold=silence_threshold,
            max_slice_duration=max_slice_duration,
            min_silence_duration=min_silence_duration,
            skip_steps=skip_steps,
        )


if __name__ == "__main__":
    run_inference()
