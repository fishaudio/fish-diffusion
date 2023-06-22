import re
import click
import hydra
from loguru import logger
from omegaconf import OmegaConf
import os
from pathlib import Path
import datetime


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True),
    default=None,
    help="input file",
    required=True,
)
@click.argument(
    "output", type=click.Path(), default=None, help="output file", required=True
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="test/test2.yaml",
    help="config name",
)
@click.option(
    "--checkpoint",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="ckpt file",
    callback=lambda ctx, param, value: value
    if value is None or value.endswith(".ckpt")
    else None,
    show_default=True,
)
@click.option("--help", "-h", is_flag=True, help="Show help")
def inference(input, output, config, checkpoint, help):
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
            from hifisinger_svc.inference import HiFiSingerSVCInference as Inference
        elif model_type == "DiffSVC":
            from tools.diffusion.inference import SVCInference as Inference
        else:
            logger.error(f"Unknown model type {model_type}")
            raise ValueError(f"Unknown model type {model_type}")
