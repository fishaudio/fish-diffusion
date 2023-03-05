from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from loguru import logger
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.archs.diffsinger.diffsinger import DiffSingerLightning
from fish_diffusion.datasets.utils import build_loader_from_config

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

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
    parser.add_argument(
        "--only-train-speaker-embeddings",
        action="store_true",
        default=False,
        help="Only train speaker embeddings.",
    )

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = DiffSingerLightning(cfg)

    # We only load the state_dict of the model, not the optimizer.
    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        result = model.load_state_dict(state_dict, strict=False)

        missing_keys = set(result.missing_keys)
        unexpected_keys = set(result.unexpected_keys)

        # Make sure incorrect keys are just noise predictor keys.
        unexpected_keys = unexpected_keys - set(
            i.replace(".naive_noise_predictor.", ".") for i in missing_keys
        )

        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        if args.only_train_speaker_embeddings:
            for name, param in model.named_parameters():
                if "speaker_encoder" not in name:
                    param.requires_grad = False

            logger.info(
                "Only train speaker embeddings, all other parameters are frozen."
            )

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
