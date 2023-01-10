import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from diff_svc.datasets.simple_dataset import SimpleDataset

from diff_svc.schedulers.lambda_warmup_cosine_scheduler import (
    LambdaWarmUpCosineScheduler,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from model.loss import DiffSingerLoss

from model.diffsinger import DiffSinger
from utils.tools import get_configs_of, synth_one_sample
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.strategies import DDPStrategy
from diff_svc.vocoders import NsfHifiGAN


class DiffSVC(pl.LightningModule):
    def __init__(self, args, preprocess_config, model_config, train_config):
        super().__init__()

        self.model = DiffSinger(args, preprocess_config, model_config, train_config)
        self.loss = DiffSingerLoss(args, preprocess_config, model_config, train_config)

        # 音频编码器, 将梅尔谱转换为音频
        self.vocoder = NsfHifiGAN()
        self.vocoder.freeze()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        lambda_func = LambdaWarmUpCosineScheduler(
            warm_up_steps=1000,
            lr_min=0.0001,
            lr_max=0.001,
            lr_start=0.0001,
            max_decay_steps=150000,
        )

        scheduler = LambdaLR(optimizer, lr_lambda=lambda_func)

        return [optimizer], [dict(scheduler=scheduler, interval="step")]

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
        losses = self.loss(batch, output)
        total_loss, mel_loss, noise_loss = losses

        self.log(f"{mode}_loss", total_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{mode}_mel_loss", mel_loss, batch_size=batch_size, sync_dist=True)
        self.log(
            f"{mode}_noise_loss",
            noise_loss,
            batch_size=batch_size,
            sync_dist=True,
        )

        if mode == "valid":
            figs, wav_reconstruction, wav_prediction = synth_one_sample(
                args,
                batch,
                pitches,
                output,
                self.vocoder,
                model_config,
                preprocess_config,
                self.model.diffusion,
            )

            # WanDB logger
            self.logger.experiment.log(
                {
                    "reconstruction_mel": [
                        wandb.Image(figs["mel"], caption="reconstruction_mel"),
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

            plt.close(figs["mel"])

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")


def get_config():
    # This should be removed after finishing editing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)

    return args, preprocess_config, model_config, train_config


if __name__ == "__main__":
    args, preprocess_config, model_config, train_config = get_config()
    model = DiffSVC(args, preprocess_config, model_config, train_config)

    train_dataset = SimpleDataset(
        "filelists/train.txt",
        preprocess_config["preprocessing"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataset = SimpleDataset(
        "filelists/test.txt",
        preprocess_config["preprocessing"],
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=0.5,
        accumulate_grad_batches=4,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        max_steps=200000,
        precision=16,
        logger=WandbLogger(project="diff-svc", log_model="all"),
        callbacks=[
            ModelCheckpoint(
                filename="diff-svc-{epoch:02d}-{valid_loss:.2f}",
                every_n_train_steps=1000,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
