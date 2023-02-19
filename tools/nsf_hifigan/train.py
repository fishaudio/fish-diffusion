import itertools
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from mmengine import Config
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fish_diffusion.datasets import DATASETS
from fish_diffusion.datasets.repeat import RepeatDataset
from fish_diffusion.utils.audio import get_mel_from_audio
from fish_diffusion.utils.viz import plot_mel
from fish_diffusion.vocoders.nsf_hifigan.models import (
    AttrDict,
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)


class HSFHifiGAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        with open(config.model.config) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        self.generator = Generator(self.h)
        self.mpd = MultiPeriodDiscriminator(self.h["discriminator_periods"])
        self.msd = MultiScaleDiscriminator()

        # Load pretrained model
        state_dict = torch.load(config.model.generator_checkpoint, map_location="cpu")
        self.generator.load_state_dict(state_dict)

        state_dict = torch.load(
            config.model.discriminator_checkpoint, map_location="cpu"
        )
        self.mpd.load_state_dict(state_dict["mpd"])
        self.msd.load_state_dict(state_dict["msd"])

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(), lr=1e-4, betas=(0.8, 0.99)
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=1e-4,
            betas=(0.8, 0.99),
        )

        state_dict = torch.load(
            self.config.model.discriminator_checkpoint, map_location="cpu"
        )
        optim_g.load_state_dict(state_dict["optim_g"])
        optim_d.load_state_dict(state_dict["optim_d"])

        return [optim_g, optim_d], []

    def training_step(self, batch, batch_idx, optimizer_idx, mode="train"):
        if optimizer_idx == 0:
            y_g_hat = self.generator(batch["mels"], batch["pitches"])
            y_g_hat_mel = get_mel_from_audio(y_g_hat.squeeze(1))

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(batch["mels"], y_g_hat_mel)

            _, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(batch["audios"], y_g_hat)
            _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(batch["audios"], y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)
            loss_gen_all = (
                loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45
            )

            self.log(
                f"{mode}_loss_gen",
                loss_gen_all,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            if mode == "valid":
                image_mels = plot_mel(
                    [
                        y_g_hat_mel.cpu().numpy(),
                        batch["mels"].cpu().numpy(),
                    ],
                    ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
                )

                self.logger.experiment.log(
                    {
                        f"reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        f"wavs": [
                            wandb.Audio(
                                batch["audios"].cpu().numpy(),
                                sample_rate=44100,
                                caption=f"gt",
                            ),
                            wandb.Audio(
                                y_g_hat.cpu().numpy(),
                                sample_rate=44100,
                                caption=f"prediction",
                            ),
                        ],
                    },
                )

                plt.close(image_mels)

            return loss_gen_all

        elif optimizer_idx == 1:
            y_g_hat = self.generator(batch["mels"], batch["pitches"])

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(batch["mels"], y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(batch["mels"], y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            self.log(
                f"{mode}_loss_disc",
                loss_disc_all,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            return loss_disc_all

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, optimizer_idx, mode="train")

    def validation_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, optimizer_idx, mode="valid")


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None, help="Wandb run id.")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = HSFHifiGAN(cfg)

    logger = WandbLogger(
        project=cfg.model.type,
        save_dir="logs",
        log_model=True,
        name=args.name,
        entity=args.entity,
        resume="must" if args.resume_id else False,
        id=args.resume_id,
    )

    trainer = pl.Trainer(
        logger=logger,
        **cfg.trainer,
    )

    train_dataset = DATASETS.build(cfg.dataset.train)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = DATASETS.build(cfg.dataset.valid)
    valid_dataset = RepeatDataset(
        valid_dataset, repeat=trainer.num_devices, collate_fn=valid_dataset.collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)
