import argparse

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from mmengine import Config
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from fish_diffusion.archs.diffsinger import DiffSinger
from fish_diffusion.datasets import DATASETS
from fish_diffusion.schedulers.cosine_scheduler import LambdaCosineScheduler
from fish_diffusion.utils.viz import viz_synth_sample
from fish_diffusion.vocoders import NsfHifiGAN


class FishDiffusion(pl.LightningModule):
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

        scheduler = StepLR(optimizer, step_size=50000, gamma=0.5)

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
                        "reconstruction_mel": wandb.Image(
                            mel_fig, caption="reconstruction_mel"
                        ),
                        "wavs": [
                            wandb.Audio(
                                wav_reconstruction.to(torch.float32).cpu().numpy(),
                                sample_rate=44100,
                                caption="reconstruction_wav (gt)",
                            ),
                            wandb.Audio(
                                wav_prediction.to(torch.float32).cpu().numpy(),
                                sample_rate=44100,
                                caption="prediction_wav",
                            ),
                        ],
                    },
                )

                plt.close(mel_fig)

        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = FishDiffusion(cfg.model)

    train_dataset = DATASETS.build(cfg.dataset.train)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = DATASETS.build(cfg.dataset.valid)

    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    trainer = pl.Trainer(
        logger=WandbLogger(
            project=cfg.model.type,
            save_dir="logs",
            log_model=True,
            entity="fish-audio",
            resume="must" if args.resume_id else False,
            id=args.resume_id,
        ),
        resume_from_checkpoint=args.resume,
        **cfg.trainer,
    )

    trainer.fit(model, train_loader, valid_loader)
