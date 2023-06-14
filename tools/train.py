import click
import hydra
from loguru import logger
from omegaconf import OmegaConf
import os
from pathlib import Path
import datetime
import re


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="test/test2.yaml",
    help="config name",
)
@click.option("--entity", "-e", type=str, default="fish-audio", help="entity for wandb")
@click.option("--tensorboard", "-t", is_flag=True, help="Log to tensorboard")
@click.option("--resume", "-r", is_flag=True, help="Resume training")
@click.option("--resume-id", "-i", type=str, default=None, help="Resume training")
@click.option(
    "--checkpoint",
    "-k",
    type=click.Path(exists=True),
    default=None,
    help="Resume training ckpt file",
    callback=lambda ctx, param, value: value
    if value is None or value.endswith(".ckpt")
    else None,
    show_default=True,
)
def main(config, entity, tensorboard, resume, resume_id, checkpoint):
    run_dir = Path(config).parent.name
    config = Path(config).stem
    logger.info(f"Running {config} in {run_dir}")
    with hydra.initialize(config_path=f"../{run_dir}", job_name=run_dir):
        cfg = hydra.compose(config_name=config)
        model = cfg.model.type
        OmegaConf.set_struct(cfg, False)  # Allow changes to the config
        # name=HIFI_SVC_ARIA entity=fish-audio tensorboard=true
        cfg.name = run_dir
        cfg.entity = entity
        cfg.tensorboard = tensorboard
        OmegaConf.set_struct(cfg, True)

        os.chdir(run_dir)
        if model == "DiffSVC":
            from tools.diffusion.train import train
        elif model == "HiFiSVC":
            from tools.hifisinger.train import train
        else:
            raise ValueError(f"Unknown model: {model}")

        if resume:
            if resume_id is None:
                # get id from logs/model/resume_id/xxx.ckpt
                resume_ids = sorted(
                    [
                        path
                        for path in (Path("logs") / model).glob("*")
                        if re.match("^[a-zA-Z0-9]+$", str(path.name))
                    ],
                    key=os.path.getctime,
                )
                if len(resume_ids) == 0:
                    raise ValueError("No resume id found")
                elif len(resume_ids) > 1:
                    # get the latest one
                    logger.warning(
                        f"Multiple resume ids found, using the latest one {resume_ids[-1]}"
                    )
                cfg.resume_id = resume_ids[-1].name
                # get the latest checkpoint from the resume id/ folder
            else:
                cfg.resume_id = resume_id

            if checkpoint is not None:
                cfg.resume = checkpoint
            else:
                ckpts = sorted(
                    (Path("logs") / model / cfg.resume_id / "checkpoints").glob(
                        "*.ckpt"
                    ),
                    key=os.path.getctime,
                )
                if len(ckpts) == 0:
                    raise ValueError("No checkpoint found")
                elif len(ckpts) > 1:
                    # get the latest one
                    logger.warning(
                        f"Multiple checkpoints found, using the latest one {ckpts[-1]}"
                    )
                cfg.resume = ckpts[-1]
        # save config for this run
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg_path = Path("logs") / model / curr_time / "config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, f"{cfg_path}")
        train(cfg)


if __name__ == "__main__":
    main()
