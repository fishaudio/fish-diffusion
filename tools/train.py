import click
import hydra
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from box import Box
import os
from pathlib import Path


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="test/test2.yaml",
    help="config name",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="hifisinger",
    help="model to train: diffusion or hifisinger",
)
@click.option("--entity", "-e", type=str, default="fish-audio", help="entity for wandb")
@click.option("--tensorboard", "-t", is_flag=True, help="Log to tensorboard")
def main(config, model, entity, tensorboard):
    run_dir = Path(config).parent.name
    config = Path(config).stem
    logger.info(f"Running {config} in {run_dir}")
    with hydra.initialize(config_path=f"../{run_dir}", job_name=run_dir):
        cfg = hydra.compose(config_name=config)
        OmegaConf.set_struct(cfg, False)  # Allow changes to the config
        # name=HIFI_SVC_ARIA entity=fish-audio tensorboard=true
        cfg.name = run_dir
        cfg.entity = entity
        cfg.tensorboard = tensorboard
        OmegaConf.set_struct(cfg, True)

        os.chdir(run_dir)
        if model == "diffusion":
            from tools.diffusion.train import train
        elif model == "hifisinger":
            from tools.hifisinger.train import train
        else:
            raise ValueError(f"Unknown model: {model}")
        train(cfg)


if __name__ == "__main__":
    main()
