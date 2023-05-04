import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
import hydra
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.archs.diffsinger import DiffSingerLightning
from fish_diffusion.datasets.utils import build_loader_from_config

# Import resolvers to register them
from tools.diffusion import resolvers
from hydra.utils import instantiate
from box import Box

torch.set_float32_matmul_precision("medium")


# python train.py --config-name svc_huber_soft name=xxxx entity=xxx
# Load the configuration file
@hydra.main(config_name=None, config_path="../../configs")
def main(cfg: DictConfig) -> None:
    from loguru import logger

    resolvers.register_resolvers(OmegaConf=OmegaConf)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = Box(cfg)

    pl.seed_everything(594461, workers=True)
    model = DiffSingerLightning(cfg)

    # We only load the state_dict of the model, not the optimizer.
    if cfg.pretrained:
        state_dict = torch.load(cfg.pretrained, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Drop vocoder.*
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("vocoder.")
        }

        result = model.load_state_dict(state_dict, strict=False)

        missing_keys = set(result.missing_keys)
        unexpected_keys = set(result.unexpected_keys)

        # Make sure incorrect keys are just noise predictor keys.
        unexpected_keys = unexpected_keys - set(
            i.replace(".naive_noise_predictor.", ".") for i in missing_keys
        )

        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        if cfg.only_train_speaker_embeddings:
            for name, param in model.named_parameters():
                if "speaker_encoder" not in name:
                    param.requires_grad = False

            logger.info(
                "Only train speaker embeddings, all other parameters are frozen."
            )

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
    callbacks = [instantiate(cb) for cb in cfg.trainer.callbacks]
    print(cfg.trainer.callbacks)
    del cfg.trainer.callbacks
    print(callbacks)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer.to_dict(),
    )

    train_loader, valid_loader = build_loader_from_config(cfg, trainer.num_devices)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
