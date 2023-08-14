from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from loguru import logger
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.archs.diffsinger import DiffSingerLightning
from fish_diffusion.datasets.utils import build_loader_from_config

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True


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

        # Drop vocoder.*
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("vocoder.")
        }

        if getattr(model, "ema_model", None) is None and any(
            k.startswith("ema_model.") for k in state_dict.keys()
        ):
            logger.warning(
                f"EMA model doesn't exist in config, drop all models and replace with ema_model."
            )
            state_dict = {
                k.replace("ema_model.", "model."): v
                for k, v in state_dict.items()
                if k.startswith("ema_model.")
            }

        # If model.speaker_encoder.embedding.weight doesn't match, we need to drop it.
        if state_dict.get("model.speaker_encoder.embedding.weight") is not None:
            if (
                state_dict["model.speaker_encoder.embedding.weight"].shape
                != model.model.speaker_encoder.embedding.weight.shape
            ):
                logger.warning(f"Speaker embedding mismatch, rebuilding from scratch.")
                del state_dict["model.speaker_encoder.embedding.weight"]

        # Do the same for the EMA model.
        if state_dict.get("ema_model.speaker_encoder.embedding.weight") is not None:
            if (
                state_dict["ema_model.speaker_encoder.embedding.weight"].shape
                != model.ema_model.speaker_encoder.embedding.weight.shape
            ):
                del state_dict["ema_model.speaker_encoder.embedding.weight"]

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
