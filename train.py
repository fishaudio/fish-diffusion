from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import soundfile as sf
import torch
import wandb
from mmengine import Config
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from fish_diffusion.archs.diffsinger import DiffSinger
from fish_diffusion.datasets import DATASETS
from fish_diffusion.datasets.repeat import RepeatDataset
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

        if mode != "valid":
            return output["loss"]

        x = self.model.diffusion.inference(output["features"])

        for idx, (gt_mel, gt_pitch, predict_mel, predict_mel_len) in enumerate(
            zip(batch["mels"], pitches, x, batch["mel_lens"])
        ):
            image_mels, wav_reconstruction, wav_prediction = viz_synth_sample(
                gt_mel=gt_mel,
                gt_pitch=gt_pitch,
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

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")


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

    model = FishDiffusion(cfg.model)

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
        resume_from_checkpoint=args.resume,
        **cfg.trainer,
    )

    train_dataset = DATASETS.build(cfg.dataset.train)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = DATASETS.build(cfg.dataset.valid)
    valid_dataset = RepeatDataset(
        valid_dataset, repeat=trainer.num_devices, collate_fn=valid_dataset.collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    # Check if dataset is empty
    assert len(train_dataset) > 0, "Train dataset is empty, please double check!"
    assert len(valid_dataset) > 0, "Valid dataset is empty, please double check!"

    trainer.fit(model, train_loader, valid_loader)
