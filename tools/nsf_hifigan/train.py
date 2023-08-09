import itertools
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from fish_diffusion.datasets.utils import build_loader_from_config
from fish_diffusion.modules.vocoders.nsf_hifigan.models import (
    AttrDict,
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from fish_diffusion.utils.audio import dynamic_range_compression, get_mel_transform
from fish_diffusion.utils.viz import plot_mel

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True


class HSFHifiGAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        with open(config.model.config) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.config = config

        self.generator = Generator(self.h)
        self.mpd = MultiPeriodDiscriminator(self.h["discriminator_periods"])
        self.msd = MultiScaleDiscriminator()

        # This is for validation only
        self.mel_transform = get_mel_transform(
            sample_rate=self.h.sampling_rate,
            n_fft=self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            f_min=self.h.fmin,
            f_max=self.h.fmax,
            n_mels=self.h.num_mels,
        )

        # The below are for training
        self.multi_scale_mels = [
            get_mel_transform(
                sample_rate=self.h.sampling_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                # Shouldn't ise fmin and fmax here
                # Otherwise high frequency will be cut off
                f_min=0,
                f_max=self.h.sampling_rate // 2,
                n_mels=self.h.num_mels,
            )
            for (n_fft, hop_length, win_length) in [
                (self.h.n_fft, self.h.hop_size, self.h.win_size),
                (2048, 270, 1080),
                (4096, 540, 2160),
            ]
        ]

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

    def generator_envelope_loss(self, y, y_hat):
        def extract_envelope(signal, kernel_size=100, stride=50):
            envelope = F.max_pool1d(signal, kernel_size=kernel_size, stride=stride)
            return envelope

        y_envelope = extract_envelope(y)
        y_hat_envelope = extract_envelope(y_hat)

        y_reverse_envelope = extract_envelope(-y)
        y_hat_reverse_envelope = extract_envelope(-y_hat)

        loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
            y_reverse_envelope, y_hat_reverse_envelope
        )

        return loss_envelope

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        pitches, y = (
            batch["pitches"].float(),
            batch["audio"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]
        mels = self.get_mels(y)[:, :, : mel_lens.max()]
        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mel_lens.max()]

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
        )

        # Generator
        optim_g.zero_grad()

        # We referenced STFT and Mel-Spectrogram loss from SingGAN
        # L1 STFT Loss
        stft_config = [
            (512, 50, 240),
            (1024, 120, 600),
            (2048, 240, 1200),
        ]

        loss_stft = 0
        for n_fft, hop_length, win_length in stft_config:
            y_stft = torch.stft(
                y.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_g_hat_stft = torch.stft(
                y_g_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_stft = torch.view_as_real(y_stft)
            y_g_hat_stft = torch.view_as_real(y_g_hat_stft)

            loss_stft += F.l1_loss(y_stft, y_g_hat_stft)

        loss_stft /= len(stft_config)

        # L1 Mel-Spectrogram Loss
        loss_mel = 0
        for mel_transform in self.multi_scale_mels:
            y_mel = self.get_mels(y, mel_transform)
            y_g_hat_mel = self.get_mels(y_g_hat, mel_transform)
            loss_mel += F.l1_loss(y_mel, y_g_hat_mel)

        loss_mel /= len(self.multi_scale_mels)

        loss_aux = 0.5 * loss_stft + loss_mel

        # L1 Envelope Loss
        loss_envelope = self.generator_envelope_loss(y, y_g_hat)
        self.log(
            "train_loss_g_envelope",
            loss_envelope,
            on_step=True,
            prog_bar=False,
            sync_dist=True,
        )

        # Discriminator Loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = (
            loss_gen_s
            + loss_gen_f
            + loss_fm_s
            + loss_fm_f
            + loss_envelope
            + loss_aux * 45
        )

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
        pitches, audios = (
            batch["pitches"].float(),
            batch["audio"].float(),
        )

        mel_lens = batch["audio_lens"] // self.config["hop_length"]

        mels = self.get_mels(audios)[:, :, : mel_lens.max()]
        y_g_hat = self.generator(mels, pitches)
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mel_lens.max()]

        # L1 Mel-Spectrogram Loss
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
                                sample_rate=self.h.sampling_rate,
                                caption=f"gt",
                            ),
                            wandb.Audio(
                                gen_audio[0, :audio_len],
                                sample_rate=self.h.sampling_rate,
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
                    sample_rate=self.h.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.h.sampling_rate,
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

    model = HSFHifiGAN(cfg)

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
