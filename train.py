import argparse
import pytorch_lightning as pl
import torch
from diff_svc.datasets.simple_dataset import SimpleDataset

from diff_svc.schedulers.cosine_scheduler import (
    LambdaCosineScheduler,
)
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.optim import AdamW

from diff_svc.archs.diffsinger import DiffSinger
from utils.tools import get_configs_of, viz_synth_sample
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.strategies import DDPStrategy
from diff_svc.vocoders import NsfHifiGAN


class DiffSVC(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffSinger(model_config)

        # 音频编码器, 将梅尔谱转换为音频
        self.vocoder = NsfHifiGAN()
        self.vocoder.freeze()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=8e-4, betas=(0.9, 0.98), eps=1e-9)

        # lambda_func = LambdaCosineScheduler(
        #     lr_min=1e-5,
        #     lr_max=8e-4,
        #     max_decay_steps=150000,
        # )

        # scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)

        scheduler = StepLR(optimizer, step_size=40000, gamma=0.5)

        return [optimizer], dict(scheduler=scheduler, interval="step")

    def _step(self, batch, batch_idx, mode):
        assert batch["pitches"].shape[1] == batch["mels"].shape[1]

        pitches = batch["pitches"].clone()
        batch_size = batch["speakers"].shape[0]

        output = self.model(
            speakers=batch["speakers"],
            contents=batch["contents"],
            src_lens=batch["content_lens"],
            max_src_len=batch["max_content_len"],
            mels=batch["mels"],
            mel_lens=batch["mel_lens"],
            max_mel_len=batch["max_mel_len"],
            pitches=batch["pitches"],
        )

        self.log(f"{mode}_loss", output["loss"], batch_size=batch_size, sync_dist=True)

        if mode == "valid":
            x = self.model.diffusion.inference(output["features"])
            for gt_mel, gt_pitch, predict_mel, predict_mel_len in zip(
                batch["mels"], pitches, x, batch["mel_lens"]
            ):
                mel_fig, wav_reconstruction, wav_prediction = viz_synth_sample(
                    gt_mel=gt_mel,
                    gt_pitch=gt_pitch,
                    predict_mel=predict_mel,
                    predict_mel_len=predict_mel_len,
                    vocoder=self.vocoder,
                )

                # WanDB logger
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": [
                            wandb.Image(mel_fig, caption="reconstruction_mel"),
                        ],
                        "target_wav": [
                            wandb.Audio(
                                wav_reconstruction.to(torch.float32).cpu().numpy(),
                                sample_rate=44100,
                                caption="reconstruction_wav",
                            ),
                        ],
                        "prediction_wav": [
                            wandb.Audio(
                                wav_prediction.to(torch.float32).cpu().numpy(),
                                sample_rate=44100,
                                caption="prediction_wav",
                            ),
                        ],
                    }
                )

                plt.close(mel_fig)

        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")


if __name__ == "__main__":
    preprocess_config, model_config, train_config = get_configs_of("ms")
    model = DiffSVC(model_config)

    train_dataset = SimpleDataset(
        preprocess_config["preprocessing"],
        "dataset/aria",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        persistent_workers=True,
        num_workers=2,
    )

    valid_dataset = SimpleDataset(
        preprocess_config["preprocessing"],
        "dataset/valid",
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=0.5,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        max_steps=160000,
        precision=16,
        logger=WandbLogger(
            project="diff-svc", save_dir="logs", log_model="all"
        ),  # resume="must", id="2qx3vhvp"),
        callbacks=[
            ModelCheckpoint(
                filename="diff-svc-{epoch:02d}-{valid_loss:.2f}",
                every_n_train_steps=1000,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        # resume_from_checkpoint="diff-svc/2qx3vhvp/checkpoints/diff-svc-epoch=571-valid_loss=0.05.ckpt"
    )

    trainer.fit(model, train_loader, valid_loader)
