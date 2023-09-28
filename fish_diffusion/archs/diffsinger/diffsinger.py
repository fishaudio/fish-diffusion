import loralib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from mmengine.optim import OPTIMIZERS
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fish_diffusion.modules.encoders import ENCODERS
from fish_diffusion.modules.vocoders import VOCODERS
from fish_diffusion.modules.vocoders.builder import VOCODERS
from fish_diffusion.schedulers import LR_SCHEUDLERS
from fish_diffusion.utils.viz import viz_synth_sample

from .diffusions import DIFFUSIONS
from .grad_tts import GradTTS


class DiffSinger(nn.Module):
    """DiffSinger"""

    def __init__(self, model_config):
        super(DiffSinger, self).__init__()

        self.text_encoder = ENCODERS.build(model_config.text_encoder)
        self.diffusion = DIFFUSIONS.build(model_config.diffusion)

        if getattr(model_config, "speaker_encoder", None):
            self.speaker_encoder = ENCODERS.build(model_config.speaker_encoder)

        if getattr(model_config, "pitch_encoder", None):
            self.pitch_encoder = ENCODERS.build(model_config.pitch_encoder)

        if getattr(model_config, "pitch_shift_encoder", None):
            self.pitch_shift_encoder = ENCODERS.build(model_config.pitch_shift_encoder)

        if getattr(model_config, "energy_encoder", None):
            self.energy_encoder = ENCODERS.build(model_config.energy_encoder)

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

        if (
            speakers is not None
            and speakers.ndim in [2, 3]
            and torch.is_floating_point(speakers)
        ):
            speaker_embed = speakers
        elif speakers is not None and hasattr(self, "speaker_encoder"):
            speaker_embed = self.speaker_encoder(speakers)
        else:
            speaker_embed = None

        if speaker_embed is not None and speaker_embed.ndim == 2:
            speaker_embed = speaker_embed[:, None, :]

        # Ignore speaker embedding for now
        if speaker_embed is not None:
            features += speaker_embed

        if hasattr(self, "pitch_encoder"):
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
            x_masks=mel_masks,
            x_lens=mel_lens,
            cond_masks=mel_masks,
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
            features["features"],
            mel,
            x_masks=features["x_masks"],
            cond_masks=features["cond_masks"],
        )

        if "loss" in features and features["loss"] is not None:
            output_dict["loss"] = output_dict["loss"] + features["loss"]

        # For validation
        output_dict["features"] = features["features"]
        output_dict["x_masks"] = features["x_masks"]
        output_dict["x_lens"] = features["x_lens"]
        output_dict["cond_masks"] = features["cond_masks"]

        return output_dict


class DiffSingerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        model_fn = GradTTS if config.model.type == "GradTTS" else DiffSinger
        self.model = model_fn(config.model)
        self.config = config
        self.ema_momentum = config.get("ema_momentum", None)
        self.lora = config.get("lora", False)
        self.lora_rank = config.get("lora_rank", 16)

        if self.lora:
            self.build_lora(self.model)

        if self.ema_momentum is not None:
            self.ema_model = model_fn(config.model)

            if self.lora:
                self.build_lora(self.ema_model)

            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()

            for param in self.ema_model.parameters():
                param.requires_grad = False

        if self.lora:
            loralib.mark_only_lora_as_trainable(self.model)

        # Vocoder, converting mel / hidden features to audio
        self.vocoder = VOCODERS.build(config.model.vocoder)
        self.vocoder.freeze()

    def build_lora(self, module):
        if not self.lora:
            return

        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    loralib.Linear(
                        child.in_features, child.out_features, self.lora_rank
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    loralib.Embedding(
                        child.num_embeddings, child.embedding_dim, self.lora_rank
                    ),
                )
            else:
                self.build_lora(child)

    def configure_optimizers(self):
        optimizer = OPTIMIZERS.build(
            {
                "params": self.parameters(),
                **self.config.optimizer,
            }
        )

        if getattr(self.config, "scheduler", None) is None:
            return optimizer

        scheduler = LR_SCHEUDLERS.build(
            {
                "optimizer": optimizer,
                **self.config.scheduler,
            }
        )

        return [optimizer], dict(scheduler=scheduler, interval="step")

    def _step(self, batch, batch_idx, mode):
        model = (
            self.ema_model
            if self.ema_momentum is not None and mode == "valid"
            else self.model
        )

        if "pitches" not in batch:
            batch["pitches"] = pitches = None
        else:
            assert batch["pitches"].shape[1] == batch["mel"].shape[1]
            pitches = batch["pitches"].clone()

        batch_size = batch["mel"].shape[0]

        output = model(
            speakers=batch.get("speaker", None),
            contents=batch["contents"],
            contents_lens=batch["contents_lens"],
            contents_max_len=batch["contents_max_len"],
            mel=batch["mel"],
            mel_lens=batch["mel_lens"],
            mel_max_len=batch["mel_max_len"],
            pitches=batch.get("pitches", None),
            pitch_shift=batch.get("key_shift", None),
            phones2mel=batch.get("phones2mel", None),
            energy=batch.get("energy", None),
        )

        self.log(
            f"{mode}_loss",
            output["loss"],
            batch_size=batch_size,
            sync_dist=True,
            prog_bar=True,
        )

        # If we need to log other metrics
        for k, v in output.get("metrics", {}).items():
            self.log(
                f"{mode}_{k}",
                v,
                batch_size=batch_size,
                sync_dist=True,
            )

        if mode != "valid":
            return output["loss"]

        x = model.diffusion(
            output["features"],
            x_masks=output["x_masks"],
            cond_masks=output["cond_masks"],
        )

        if pitches is None:
            pitches = [None] * batch_size

        for idx, (
            gt_mel,
            gt_pitch,
            predict_mel,
            predict_mel_len,
            gt_mel_len,
        ) in enumerate(
            zip(batch["mel"], pitches, x, output["x_lens"], batch["mel_lens"])
        ):
            image_mels, wav_reconstruction, wav_prediction = viz_synth_sample(
                gt_mel=gt_mel,
                gt_pitch=gt_pitch[:, 0] if gt_pitch is not None else None,
                predict_mel=predict_mel,
                predict_mel_len=predict_mel_len,
                gt_mel_len=gt_mel_len,
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
        loss = self._step(batch, batch_idx, mode="train")

        if self.ema_momentum is None:
            return loss

        # Update EMA
        ema_params_list, params_list = [], []
        for ema_param, param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_params_list.append(ema_param.data)
            params_list.append(param.data)

        # Update student_ema
        torch._foreach_mul_(ema_params_list, self.ema_momentum)
        torch._foreach_add_(ema_params_list, params_list, alpha=1.0 - self.ema_momentum)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")
