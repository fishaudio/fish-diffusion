import click
import hydra
from loguru import logger
from matplotlib.pyplot import flag
from omegaconf import OmegaConf, DictConfig
from box import Box
import os


@click.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--config", "-c", type=str, default="config", help="config name")
@click.option(
    "--model",
    "-m",
    type=str,
    default="diffusion",
    help="model to train: diffusion or hifisinger",
)
@click.option("--entity", "-e", type=str, default="fish-audio", help="entity for wandb")
@click.option("--tensorboard", "-t", is_flag=True, help="Log to tensorboard")
def main(run_dir, config, model, entity, tensorboard):
    with hydra.initialize(config_path=f"./{run_dir}", job_name=run_dir):
        cfg = hydra.compose(config_name=config)
        original_cwd = os.getcwd()
        OmegaConf.set_struct(cfg, False)  # Allow changes to the config
        cfg.project_root = original_cwd
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
        train(cfg)


if __name__ == "__main__":
    main()
