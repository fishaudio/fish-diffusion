import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
import hydra
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from fish_diffusion.archs.hifisinger import HiFiSingerV1Lightning, HiFiSingerV2Lightning
from fish_diffusion.datasets.utils import build_loader_from_config
from hydra.utils import instantiate
from box import Box

from hydra.utils import get_method

torch.set_float32_matmul_precision("medium")


# python train.py --config-name svc_hifi name=xxxx entity=xxx
# Load the configuration file
@hydra.main(config_name=None, config_path="../../configs")
def train(cfg: DictConfig) -> None:
    from loguru import logger as loguru_logger

    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    cfg = Box(cfg)  # type: ignore

    pl.seed_everything(594461, workers=True)
    if cfg.model.encoder._target_.lower().split(".")[-1] == "RefineGAN".lower():
        model = HiFiSingerV2Lightning(cfg)
    elif cfg.model.encoder._target_.lower().split(".")[-1] == "HiFiGAN".lower():
        model = HiFiSingerV1Lightning(cfg)
    else:
        raise NotImplementedError(f"Unknown encoder type: {cfg.model.encoder.type}")

    # We only load the state_dict of the model, not the optimizer.
    if cfg.pretrained:
        state_dict = torch.load(cfg.pretrained, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        result = model.load_state_dict(state_dict, strict=False)

        missing_keys = set(result.missing_keys)
        unexpected_keys = set(result.unexpected_keys)

        missing_keys.remove("generator.speaker_encoder.embedding.weight")

        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

    logger = (
        TensorBoardLogger("logs", name=cfg.model.type)
        if cfg.tensorboard
        else WandbLogger(
            project=cfg.model.type,
            save_dir="logs",
            log_model=True,
            name=cfg.name,
            entity=cfg.entity,
            resume="must" if cfg.resume_id else False,
            id=cfg.resume_id,
        )
    )

    if cfg.trainer.strategy is None:
        del cfg.trainer.strategy
    else:
        loguru_logger.debug(f"Using strategy: {cfg.trainer.strategy}")
        cfg.trainer.strategy.ddp_comm_hook = get_method(
            cfg.trainer.strategy.ddp_comm_hook
        )
        cfg.trainer.strategy = instantiate(cfg.trainer.strategy)
    callbacks = [instantiate(cb) for cb in cfg.trainer.callbacks]
    del cfg.trainer.callbacks
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer.to_dict(),
    )

    train_loader, valid_loader = build_loader_from_config(cfg, trainer.num_devices)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    train()
