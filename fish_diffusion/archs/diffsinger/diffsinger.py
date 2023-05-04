import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.nn import functional as F

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.vocoders import VOCODERS
from fish_diffusion.modules.vocoders.builder import VOCODERS
from fish_diffusion.schedulers import LR_SCHEUDLERS
from fish_diffusion.utils.viz import viz_synth_sample

from .diffusions import DIFFUSIONS

from hydra.utils import instantiate

from omegaconf import DictConfig


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.text_encoder = instantiate(model_config.text_encoder)
        self.diffusion = instantiate(model_config.diffusion)
        self.speaker_encoder = instantiate(model_config.speaker_encoder)
        self.pitch_encoder = instantiate(model_config.pitch_encoder)

        if "pitch_shift_encoder" in model_config:
            self.pitch_shift_encoder = instantiate(model_config.pitch_shift_encoder)

        if "energy_encoder" in model_config:
            self.energy_encoder = instantiate(model_config.energy_encoder)

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
        mel_lens=None,
        mel_max_len=None,
        pitches=None,
        pitch_shift=None,
        phones2mel=None,
        energy=None,
    ):
        src_masks = (
            self.get_mask_from_lengths(contents_lens, contents_max_len)
            if contents_lens is not None
            else None
        )

        mel_masks = (
            self.get_mask_from_lengths(mel_lens, mel_max_len)
            if mel_lens is not None
            else None
        )

        features = self.text_encoder(contents, src_masks)

        if phones2mel is not None:
            phones2mel = (
                phones2mel.unsqueeze(-1).repeat([1, 1, features.shape[-1]]).long()
            )
            features = torch.gather(features, 1, phones2mel) * (
                1 - mel_masks[:, :, None].float()
            )

        if speakers.ndim in [2, 3] and torch.is_floating_point(speakers):
            speaker_embed = speakers
        else:
            speaker_embed = self.speaker_encoder(speakers)

        if speaker_embed.ndim == 2:
            speaker_embed = speaker_embed[:, None, :]

        features += speaker_embed
        features += self.pitch_encoder(pitches)

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

        return dict(
            features=features,
            src_masks=src_masks,
            mel_masks=mel_masks,
        )

    def forward(
        self,
        speakers,
        contents,
        contents_lens,
        contents_max_len,
        mel=None,
        mel_lens=None,
        mel_max_len=None,
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
            mel_lens=mel_lens,
            mel_max_len=mel_max_len,
            pitches=pitches,
            pitch_shift=pitch_shift,
            phones2mel=phones2mel,
            energy=energy,
        )

        output_dict = self.diffusion.train_step(
            features["features"], mel, features["mel_masks"]
        )

        # For validation
        output_dict["features"] = features["features"]

        return output_dict


class DiffSingerLightning(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.model = DiffSinger(config.model)
        self.config = config

        # 音频编码器, 将梅尔谱转换为音频
        self.vocoder = instantiate(config.model.vocoder)
        self.vocoder.freeze()

    def configure_optimizers(self):
        # Instantiate the optimizer
        optimizer = instantiate(self.config.optimizer, params=self.parameters())

        # Instantiate the scheduler
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)

        return [optimizer], dict(scheduler=scheduler, interval="step")
        # optimizer = OPTIMIZERS.build(
        #     {
        #         "params": self.parameters(),
        #         **self.config.optimizer,
        #     }
        # )

        # scheduler = LR_SCHEUDLERS.build(
        #     {
        #         "optimizer": optimizer,
        #         **self.config.scheduler,
        #     }
        # )

        # return [optimizer], dict(scheduler=scheduler, interval="step")

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
            phones2mel=batch.get("phones2mel", None),
            energy=batch.get("energy", None),
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
