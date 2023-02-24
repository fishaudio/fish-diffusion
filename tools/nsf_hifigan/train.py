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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import MelSpectrogram

from fish_diffusion.datasets import RepeatDataset, VOCODERDataset
from fish_diffusion.utils.audio import dynamic_range_compression
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
        self.config = config

        self.generator = Generator(self.h)
        self.mpd = MultiPeriodDiscriminator(self.h["discriminator_periods"])
        self.msd = MultiScaleDiscriminator()

        self.mel_transform = self.get_mel_transform()

        self.automatic_optimization = False

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.h.learning_rate,
            betas=(self.h.adam_b1, self.h.adam_b2),
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=self.h.learning_rate,
            betas=(self.h.adam_b1, self.h.adam_b2),
        )

        scheduler_g = ExponentialLR(optim_g, self.h.lr_decay)
        scheduler_d = ExponentialLR(optim_d, self.h.lr_decay)

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        mels, pitches, y = (
            batch["mels"].float(),
            batch["pitches"].float(),
            batch["audios"].float(),
        )

        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mels.shape[2]]
        optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        self.manual_backward(loss_disc_all)
        optim_d.step()

        self.log(
            f"train_loss_disc",
            loss_disc_all,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Generator
        optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(mels, y_g_hat_mel)

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45

        self.manual_backward(loss_gen_all)
        optim_g.step()

        self.log(
            f"train_loss_gen",
            loss_gen_all,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

    def get_mel_transform(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=40,
        f_max=16000,
        n_mels=128,
        center=True,
        power=1.0,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
    ) -> torch.Tensor:
        transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=center,
            power=power,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )

        return transform

    def get_mels(self, audios):
        x = self.mel_transform(audios.squeeze(1))
        return dynamic_range_compression(x)

    def validation_step(self, batch, batch_idx):
        mels, pitches, audios = (
            batch["mels"].float(),
            batch["pitches"].float(),
            batch["audios"].float(),
        )

        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mels.shape[2]]

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(mels, y_g_hat_mel)
        self.log(
            "valid_loss",
            loss_mel,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if isinstance(self.logger, WandbLogger):
            for mel, gen_mel, audio, gen_audio in zip(
                mels.cpu().numpy(),
                y_g_hat_mel.cpu().numpy(),
                audios.cpu().type(torch.float32).numpy(),
                y_g_hat.type(torch.float32).cpu().numpy(),
            ):
                # exit()
                image_mels = plot_mel(
                    [
                        gen_mel,
                        mel,
                    ],
                    ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
                )

                self.logger.experiment.log(
                    {
                        f"reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        f"wavs": [
                            wandb.Audio(
                                audio[0],
                                sample_rate=self.h.sampling_rate,
                                caption=f"gt",
                            ),
                            wandb.Audio(
                                gen_audio[0],
                                sample_rate=self.h.sampling_rate,
                                caption=f"prediction",
                            ),
                        ],
                    },
                )

                plt.close(image_mels)


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

    dataset = VOCODERDataset("/mnt/nvme0/diff-wave-data")
    train_dataset, valid_dataset = random_split(
        dataset, [int(len(dataset) * 0.99), len(dataset) - int(len(dataset) * 0.99)]
    )

    train_loader = DataLoader(
        train_dataset,
        # collate_fn=dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = RepeatDataset(
        valid_dataset,
        repeat=trainer.num_devices,  # collate_fn=dataset.collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        # collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)
