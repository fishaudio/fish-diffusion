import itertools

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.vocoders.nsf_hifigan.models import AttrDict, Generator
from fish_diffusion.modules.vocoders.refinegan.generator import RefineGANGenerator
from fish_diffusion.modules.vocoders.refinegan.mpd import MultiPeriodDiscriminator
from fish_diffusion.modules.vocoders.refinegan.mrd import MultiResolutionDiscriminator
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

        if "type" in model_config.encoder and model_config.encoder.type == "RefineGAN":
            self.encoder_type = "RefineGAN"
            model_config.encoder.pop("type")
            self.encoder = RefineGANGenerator(**model_config.encoder)
        else:
            self.encoder_type = "HiFiGAN"
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

        if speakers.ndim in [2, 3] and torch.is_floating_point(speakers):
            speaker_embed = speakers
        else:
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

        if self.encoder_type == "RefineGAN":
            return self.encoder(
                features["features"].transpose(1, 2), pitches.transpose(1, 2)
            )
        else:
            return self.encoder(features["features"].transpose(1, 2), pitches[:, :, 0])


class HiFiSingerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.h = AttrDict(config.model.encoder)
        self.config = config

        self.generator = HiFiSinger(config.model)
        self.mpd = MultiPeriodDiscriminator(**config.model.mpd)
        self.mrd = MultiResolutionDiscriminator(**config.model.mrd)

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
                "params": self.generator.parameters(),
                **self.config.optimizer,
            }
        )
        optim_d = OPTIMIZERS.build(
            {
                "params": itertools.chain(self.mrd.parameters(), self.mpd.parameters()),
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

    def generator_mel_loss(self, y, y_hat):
        loss_mel = []

        for mel_transform in self.multi_scale_mels:
            y_mel = self.get_mels(y, mel_transform)
            y_g_hat_mel = self.get_mels(y_hat, mel_transform)
            loss_mel.append(F.smooth_l1_loss(y_mel, y_g_hat_mel))

        return sum(loss_mel) / len(loss_mel)

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

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    def generator_adv_loss(self, disc_outputs):
        losses = []

        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            losses.append(l)

        return sum(losses) / len(losses)

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        losses = []

        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            losses.append((r_loss + g_loss) / 2)

        return sum(losses) / len(losses)

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

        # Discriminator Loss
        optim_d.zero_grad()

        # MPD Loss
        y_g_hat_x, _ = self.mpd(y_g_hat.detach())
        y_x, _ = self.mpd(y)
        loss_mpd = self.discriminator_loss(y_x, y_g_hat_x)
        self.log(
            "train_loss_d_mpd", loss_mpd, on_step=True, prog_bar=False, sync_dist=True
        )

        # MRD Loss
        y_g_hat_x, _ = self.mrd(y_g_hat.detach())
        y_x, _ = self.mrd(y)
        loss_mrd = self.discriminator_loss(y_x, y_g_hat_x)
        self.log(
            "train_loss_d_mrd", loss_mrd, on_step=True, prog_bar=False, sync_dist=True
        )

        # Discriminator Loss
        loss_d = loss_mpd + loss_mrd
        self.log("train_loss_d", loss_d, on_step=True, prog_bar=True, sync_dist=True)

        self.manual_backward(loss_d)
        optim_d.step()

        # Generator Loss
        optim_g.zero_grad()

        # Correct the length
        corrected_length = min(y.shape[-1], y_g_hat.shape[-1])
        y = y[..., :corrected_length]
        y_g_hat = y_g_hat[..., :corrected_length]

        # L2 Mel-Spectrogram Loss
        loss_mel = self.generator_mel_loss(y, y_g_hat)
        self.log(
            "train_loss_g_mel", loss_mel, on_step=True, prog_bar=False, sync_dist=True
        )

        # L1 Envelope Loss
        loss_envelope = self.generator_envelope_loss(y, y_g_hat)
        self.log(
            "train_loss_g_envelope",
            loss_envelope,
            on_step=True,
            prog_bar=False,
            sync_dist=True,
        )

        # MPD Loss
        y_g_hat_x, _ = self.mpd(y_g_hat)
        loss_mpd = self.generator_adv_loss(y_g_hat_x)
        self.log(
            "train_loss_g_mpd", loss_mpd, on_step=True, prog_bar=False, sync_dist=True
        )

        # MRD Loss
        y_g_hat_x, _ = self.mrd(y_g_hat)
        loss_mrd = self.generator_adv_loss(y_g_hat_x)
        self.log(
            "train_loss_g_mrd", loss_mrd, on_step=True, prog_bar=False, sync_dist=True
        )

        # Generator Loss
        loss_g = 45 * loss_mel + loss_envelope + loss_mpd + loss_mrd
        self.log("train_loss_g", loss_g, on_step=True, prog_bar=True, sync_dist=True)

        self.manual_backward(loss_g)
        optim_g.step()

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
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for idx, (mel, gen_mel, audio, gen_audio, mel_len, audio_len) in enumerate(
            zip(
                mels.cpu().numpy(),
                y_g_hat_mel.cpu().numpy(),
                batch["audio"].cpu().type(torch.float32).numpy(),
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
                    sample_rate=self.config["sampling_rate"],
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.config["sampling_rate"],
                )

            plt.close(image_mels)
