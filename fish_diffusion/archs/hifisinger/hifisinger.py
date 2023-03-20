import itertools

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.vocoders.nsf_hifigan.models import (
    AttrDict,
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from fish_diffusion.schedulers import LR_SCHEUDLERS
from fish_diffusion.utils.audio import dynamic_range_compression, get_mel_transform
from fish_diffusion.utils.viz import plot_mel


class HiFiSinger(nn.Module):
    """HiFiSinger"""

    def __init__(self, model_config):
        super(HiFiSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)

        if "pitch_shift_encoder" in model_config:
            self.pitch_shift_encoder = ENCODERS.build(model_config.pitch_shift_encoder)

        if "energy_encoder" in model_config:
            self.energy_encoder = ENCODERS.build(model_config.energy_encoder)

        self.feature_fuser = nn.Sequential(
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.SiLU(),
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.SiLU(),
        )

        self.encoder = Generator(AttrDict(model_config.encoder))

    @staticmethod
    def get_mask_from_lengths(lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

    def forward_features(
        self,
        speakers,
        contents,
        contents_lens,
        contents_max_len,
        pitch_shift=None,
        phones2mel=None,
        energy=None,
    ):
        src_masks = (
            self.get_mask_from_lengths(contents_lens, contents_max_len)
            if contents_lens is not None
            else None
        )

        features = self.text_encoder(contents, src_masks)

        if phones2mel is not None:
            phones2mel = (
                phones2mel.unsqueeze(-1).repeat([1, 1, features.shape[-1]]).long()
            )
            features = torch.gather(features, 1, phones2mel) * (
                1 - src_masks[:, :, None].float()
            )

        speaker_embed = self.speaker_encoder(speakers)
        if speaker_embed.ndim == 2:
            speaker_embed = speaker_embed[:, None, :]

        features += speaker_embed

        if pitch_shift is not None and hasattr(self, "pitch_shift_encoder"):
            pitch_shift_embed = self.pitch_shift_encoder(pitch_shift)

            if pitch_shift_embed.ndim == 2:
                pitch_shift_embed = pitch_shift_embed[:, None, :]

            features += pitch_shift_embed

        if energy is not None and hasattr(self, "energy_encoder"):
            energy_embed = self.energy_encoder(energy)

            if energy_embed.ndim == 2:
                energy_embed = energy_embed[:, None, :]

            features += energy_embed

        features = self.feature_fuser(features)
        features *= 1 - src_masks[:, :, None].float()

        return dict(
            features=features,
            src_masks=src_masks,
        )

    def forward(
        self,
        speakers,
        contents,
        contents_lens,
        contents_max_len,
        pitches=None,
        pitch_shift=None,
        phones2mel=None,
        energy=None,
    ):
        features = self.forward_features(
            speakers=speakers,
            contents=contents,
            contents_lens=contents_lens,
            contents_max_len=contents_max_len,
            pitch_shift=pitch_shift,
            phones2mel=phones2mel,
            energy=energy,
        )

        return self.encoder(features["features"].transpose(1, 2), pitches[:, :, 0])


class HiFiSingerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.h = AttrDict(config.model.encoder)
        self.config = config

        self.generator = HiFiSinger(config.model)
        self.mpd = MultiPeriodDiscriminator(self.h["discriminator_periods"])
        self.msd = MultiScaleDiscriminator()

        # This is for validation only
        self.mel_transform = get_mel_transform(
            sample_rate=config.sampling_rate,
            n_fft=self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            f_min=self.h.fmin,
            f_max=self.h.fmax,
            n_mels=self.h.num_mels,
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
                n_mels=self.h.num_mels,
            )
            for (n_fft, hop_length, win_length) in self.h.multi_scale_mels
        ]

        self.automatic_optimization = False

    def configure_optimizers(self):
        optim_g = OPTIMIZERS.build(
            {
                "params": self.generator.parameters(),
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

        y_g_hat = self.generator(
            speakers=batch["speaker"],
            contents=batch["contents"],
            contents_lens=batch["contents_lens"],
            contents_max_len=batch["contents_max_len"],
            pitches=batch["pitches"],
            pitch_shift=batch.get("key_shift", None),
            phones2mel=batch.get("phones2mel", None),
            energy=batch.get("energy", None),
        )

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
            batch_size=batch["contents"].shape[0],
        )

        # Generator
        optim_g.zero_grad()

        # We referenced STFT and Mel-Spectrogram loss from SingGAN
        # L1 STFT Loss

        loss_stft = 0
        for n_fft, hop_length, win_length in self.h.multi_scale_stfts:
            y_stft = torch.stft(
                y.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_g_hat_stft = torch.stft(
                y_g_hat.squeeze(1), n_fft, hop_length, win_length, return_complex=True
            )
            y_stft = torch.view_as_real(y_stft)
            y_g_hat_stft = torch.view_as_real(y_g_hat_stft)

            loss_stft += F.l1_loss(y_stft, y_g_hat_stft)

        loss_stft /= len(self.h.multi_scale_stfts)

        # L1 Mel-Spectrogram Loss
        loss_mel = 0
        for mel_transform in self.multi_scale_mels:
            y_mel = self.get_mels(y, mel_transform)
            y_g_hat_mel = self.get_mels(y_g_hat, mel_transform)
            loss_mel += F.l1_loss(y_mel, y_g_hat_mel)

        loss_mel /= len(self.multi_scale_mels)

        loss_aux = 0.5 * loss_stft + loss_mel

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
            batch_size=batch["contents"].shape[0],
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
        y_g_hat = self.generator(
            speakers=batch["speaker"],
            contents=batch["contents"],
            contents_lens=batch["contents_lens"],
            contents_max_len=batch["contents_max_len"],
            pitches=batch["pitches"],
            pitch_shift=batch.get("key_shift", None),
            phones2mel=batch.get("phones2mel", None),
            energy=batch.get("energy", None),
        )

        mels = self.get_mels(batch["audio"])
        y_g_hat_mel = self.get_mels(y_g_hat)[:, :, : mels.shape[2]]

        # L1 Mel-Spectrogram Loss
        mel_lens = batch["contents_lens"]

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
            batch_size=batch["contents"].shape[0],
        )

        for idx, (mel, gen_mel, audio, gen_audio, mel_len, audio_len) in enumerate(
            zip(
                mels.cpu().numpy(),
                y_g_hat_mel.cpu().numpy(),
                batch["audio"].cpu().type(torch.float32).numpy(),
                y_g_hat.type(torch.float32).cpu().numpy(),
                batch["contents_lens"].cpu().numpy(),
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
