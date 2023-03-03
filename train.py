from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from mmengine import Config
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.archs.diffsinger import DiffSinger
from fish_diffusion.datasets.utils import build_loader_from_config
from fish_diffusion.utils.scheduler import LR_SCHEUDLERS
from fish_diffusion.utils.viz import viz_synth_sample
from fish_diffusion.vocoders import VOCODERS

torch.set_float32_matmul_precision("medium")


class FishDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffSinger(config.model)
        self.config = config

        # 音频编码器, 将梅尔谱转换为音频
        self.vocoder = VOCODERS.build(config.model.vocoder)
        self.vocoder.freeze()

    def configure_optimizers(self):
        self.config.optimizer.params = self.parameters()
        optimizer = OPTIMIZERS.build(self.config.optimizer)

        self.config.scheduler.optimizer = optimizer
        scheduler = LR_SCHEUDLERS.build(self.config.scheduler)

        return [optimizer], dict(scheduler=scheduler, interval="step")

    def _step(self, batch, batch_idx, mode):
        assert batch["pitches"].shape[1] == batch["mel"].shape[1]

        pitches = batch["pitches"].clone()
        batch_size = batch["speaker"].shape[0]

        output = self.model(
            speakers=batch["speaker"],
            contents=batch["contents"],
            contents_lens=batch["contents_lens"],
            contents_max_len=batch["contents_max_len"],
            mel=batch["mel"],
            mel_lens=batch["mel_lens"],
            mel_max_len=batch["mel_max_len"],
            pitches=batch["pitches"],
            pitch_shift=batch.get("key_shift", None),
        )

        self.log(f"{mode}_loss", output["loss"], batch_size=batch_size, sync_dist=True)

        if mode != "valid":
            return output["loss"]

        x = self.model.diffusion(output["features"])

        for idx, (gt_mel, gt_pitch, predict_mel, predict_mel_len) in enumerate(
            zip(batch["mel"], pitches, x, batch["mel_lens"])
        ):
            image_mels, wav_reconstruction, wav_prediction = viz_synth_sample(
                gt_mel=gt_mel,
                gt_pitch=gt_pitch[:, 0],
                predict_mel=predict_mel,
                predict_mel_len=predict_mel_len,
                vocoder=self.vocoder,
                return_image=False,
            )

            wav_reconstruction = wav_reconstruction.to(torch.float32).cpu().numpy()
            wav_prediction = wav_prediction.to(torch.float32).cpu().numpy()

            # WanDB logger
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        f"reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        f"wavs": [
                            wandb.Audio(
                                wav_reconstruction,
                                sample_rate=44100,
                                caption=f"reconstruction (gt)",
                            ),
                            wandb.Audio(
                                wav_prediction,
                                sample_rate=44100,
                                caption=f"prediction",
                            ),
                        ],
                    },
                )

            # TensorBoard logger
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    wav_reconstruction,
                    self.global_step,
                    sample_rate=44100,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    wav_prediction,
                    self.global_step,
                    sample_rate=44100,
                )

            if isinstance(image_mels, plt.Figure):
                plt.close(image_mels)

        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")


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

    model = FishDiffusion(cfg)

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

        assert len(unexpected_keys) == 0

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
