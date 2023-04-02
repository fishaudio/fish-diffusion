import itertools
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from mmengine import Config
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F

from fish_diffusion.datasets.utils import build_loader_from_config
from fish_diffusion.modules.vocoders.auto_vocoder.models import (
    Decoder,
    Encoder,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from fish_diffusion.schedulers import LR_SCHEUDLERS
from fish_diffusion.utils.audio import dynamic_range_compression, get_mel_transform
from fish_diffusion.utils.viz import plot_mel

torch.set_float32_matmul_precision("medium")


class AutoVocoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        self.encoder = Encoder(**config.model.encoder)
        self.decoder = Decoder(**config.model.decoder)

        self.mpd = MultiPeriodDiscriminator(config.model.discriminator_periods)
        self.msd = MultiScaleDiscriminator()

        # This is for validation only
        self.mel_transform = get_mel_transform(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            f_min=0,
            f_max=config.sampling_rate // 2,
            n_mels=config.num_mels,
        )

        # The bellow are for training
        self.multi_scale_mels = [
            get_mel_transform(
                sample_rate=config.sampling_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                # Shouldn't ise fmin and fmax here
                # Otherwise high frequency will be cut off
                f_min=0,
                f_max=config.sampling_rate // 2,
                n_mels=config.num_mels,
            )
            for (n_fft, hop_length, win_length) in config.model.multi_scale_mels
        ]

        self.automatic_optimization = False

    def configure_optimizers(self):
        optim_g = OPTIMIZERS.build(
            {
                "params": itertools.chain(
                    self.encoder.parameters(), self.decoder.parameters()
                ),
                **self.config.optimizer,
            }
        )
        optim_d = OPTIMIZERS.build(
            {
                "params": itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                **self.config.optimizer,
            }
        )

        scheduler_g = LR_SCHEUDLERS.build(
            {
                "optimizer": optim_g,
                **self.config.scheduler,
            }
        )
        scheduler_d = LR_SCHEUDLERS.build(
            {
                "optimizer": optim_d,
                **self.config.scheduler,
            }
        )

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        y = batch["audio"].float()

        y_g_hat = self.decoder(self.encoder(y[:, 0]))[:, None]
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
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["audio"].shape[0],
        )

        # Generator
        optim_g.zero_grad()

        # We referenced STFT and Mel-Spectrogram loss from SingGAN
        # L1 STFT Loss

        loss_stft = 0
        for n_fft, hop_length, win_length in self.config.model.multi_scale_stfts:
            y_stft = torch.stft(
                y.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_g_hat_stft = torch.stft(
                y_g_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_stft = torch.view_as_real(y_stft)
            y_g_hat_stft = torch.view_as_real(y_g_hat_stft)

            loss_stft += F.l1_loss(y_stft, y_g_hat_stft)

        loss_stft /= len(self.config.model.multi_scale_stfts)

        # L1 Mel-Spectrogram Loss
        loss_mel = 0
        for mel_transform in self.multi_scale_mels:
            y_mel = self.get_mels(y, mel_transform)
            y_g_hat_mel = self.get_mels(y_g_hat, mel_transform)
            loss_mel += F.l1_loss(y_mel, y_g_hat_mel)

        loss_mel /= len(self.multi_scale_mels)

        loss_aux = 0.5 * loss_stft + loss_mel

        # L2 Time Domain Loss
        # loss_l2 = F.mse_loss(y, y_g_hat) * 100

        # Discriminator Loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_aux * 45

        self.manual_backward(loss_gen_all)
        optim_g.step()

        self.log(
            f"train_loss_gen",
            loss_gen_all,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["audio"].shape[0],
        )

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()

        if self.trainer.is_last_batch:
            scheduler_g.step()
            scheduler_d.step()

    def get_mels(self, audios, transform=None):
        if transform is None:
            transform = self.mel_transform

        transform = transform.to(audios.device)
        x = transform(audios.squeeze(1))

        return dynamic_range_compression(x)

    def validation_step(self, batch, batch_idx):
        audios = batch["audio"].float()

        mels = self.get_mels(audios)
        y_g_hat = self.decoder(self.encoder(audios[:, 0]))[:, None]
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mels.shape[2]]

        # L1 Mel-Spectrogram Loss
        mel_lens = batch["audio_lens"] // self.config.hop_length
        # create mask
        mask = (
            torch.arange(mels.shape[2], device=mels.device)[None, :] < mel_lens[:, None]
        )
        mask = mask[:, None].float()

        loss_mel = F.l1_loss(mels * mask, y_g_hat_mel * mask)
        self.log(
            "valid_loss",
            loss_mel,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["audio"].shape[0],
        )

        for idx, (mel, gen_mel, audio, gen_audio, mel_len, audio_len) in enumerate(
            zip(
                mels.cpu().numpy(),
                y_g_hat_mel.cpu().numpy(),
                audios.cpu().type(torch.float32).numpy(),
                y_g_hat.type(torch.float32).cpu().numpy(),
                mel_lens.cpu().numpy(),
                batch["audio_lens"].cpu().numpy(),
            )
        ):
            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        f"reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        f"wavs": [
                            wandb.Audio(
                                audio[0, :audio_len],
                                sample_rate=self.config.sampling_rate,
                                caption=f"gt",
                            ),
                            wandb.Audio(
                                gen_audio[0, :audio_len],
                                sample_rate=self.config.sampling_rate,
                                caption=f"prediction",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.config.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.config.sampling_rate,
                )

            plt.close(image_mels)


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

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = AutoVocoder(cfg)

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
