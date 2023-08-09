from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from loguru import logger
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.archs.hifisinger import HiFiSingerV1Lightning, HiFiSingerV2Lightning
from fish_diffusion.datasets.utils import build_loader_from_config

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    pl.seed_everything(594461, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Use tensorboard logger, default is wandb.",
    )
    parser.add_argument("--resume-id", type=str, default=None, help="Wandb run id.")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name.")
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Pretrained model."
    )

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    if cfg.model.encoder.type.lower() == "RefineGAN".lower():
        model = HiFiSingerV2Lightning(cfg)
    elif cfg.model.encoder.type.lower() == "HiFiGAN".lower():
        model = HiFiSingerV1Lightning(cfg)
    else:
        raise NotImplementedError(f"Unknown encoder type: {cfg.model.encoder.type}")

    # We only load the state_dict of the model, not the optimizer.
    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location="cpu")
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
        if args.tensorboard
        else WandbLogger(
            project=cfg.model.type,
            save_dir="logs",
            log_model=True,
            name=args.name,
            entity=args.entity,
            resume="must" if args.resume_id else False,
            id=args.resume_id,
        )
    )

    trainer = pl.Trainer(
        logger=logger,
        **cfg.trainer,
    )

    train_loader, valid_loader = build_loader_from_config(cfg, trainer.num_devices)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)
