import random
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from textgrid import TextGrid
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from whisper import log_mel_spectrogram, pad_or_trim

from fish_diffusion.modules.feature_extractors.whisper import AlignedWhisper

phonemes = []
for i in open("dictionaries/opencpop-strict.txt"):
    _, phones = i.strip().split("\t")
    for j in phones.split():
        if j not in phonemes:
            phonemes.append(j)

phonemes = ["<PAD>", "<EOS>", "<UNK>", "AP", "SP"] + sorted(phonemes)


class WhisperDataset(Dataset):
    def __init__(self, path="dataset/mfa-data", split="train"):
        self.path = Path(path)
        self.files = sorted(list(self.path.glob("**/*.TextGrid.opt")))
        self.split = split

        random.Random(42).shuffle(self.files)

        if split == "train":
            self.files = self.files[: int(len(self.files) * 0.98)]
        elif split == "val":
            self.files = self.files[int(len(self.files) * 0.98) :]
        else:
            raise ValueError("Invalid split")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        text_grid_file = self.files[idx]
        audio, sr = librosa.load(
            str(text_grid_file).replace(".TextGrid.opt", ".wav"), sr=16000
        )

        # Augment: Time stretch + pitch shift
        if self.split == "train":
            speed_up_ratio = random.randint(80, 120) / 100
            audio = librosa.effects.time_stretch(audio, rate=speed_up_ratio)

            pitch_shift = random.randint(-3, 3)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)

            audio += np.random.normal(0, 0.005, audio.shape) * np.amax(audio)
        else:
            speed_up_ratio = 1

        mel = log_mel_spectrogram(audio)

        mel_len = mel.shape[1]
        feature_len = mel_len // 2
        mel = pad_or_trim(mel, 3000)

        grid = TextGrid.fromFile(str(text_grid_file))
        phones = [i.mark for i in grid.tiers[1]]

        durations = [(i.minTime, i.maxTime) for i in grid.tiers[1]]
        durations = (
            (torch.tensor(durations, dtype=torch.float) * 50 * (1 / speed_up_ratio))
            .round()
            .long()
        )
        aligned_phones = torch.zeros(
            (3000 // 2,), dtype=torch.long
        )  # All uncovered frames are ignored

        for i, (start, end) in enumerate(durations):
            phone = phones[i]

            if phone in phonemes:
                aligned_phones[start:end] = phonemes.index(phone)

        return {
            "mel": mel,
            "mel_len": mel_len,
            "phones": aligned_phones,
            "phones_len": feature_len,
        }


class WhisperModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.model = AlignedWhisper.load("medium", len(phonemes), n_outputs=256)

    def _step(self, batch, batch_idx, mode="train"):
        mels = batch["mel"]
        phones = batch["phones"]

        mask = phones != 0

        audio_embeddings = self.model.forward_audio(mels)
        phones_embeddings = self.model.forward_phones(phones)

        embedding_loss = F.l1_loss(
            audio_embeddings[mask], phones_embeddings[mask]
        ).mean()

        audio_aux = self.model.forward_decoder(audio_embeddings)
        loss1 = F.cross_entropy(audio_aux.transpose(1, 2), phones)

        self.log(
            f"{mode}_embedding_loss", embedding_loss, prog_bar=True, sync_dist=True
        )
        self.log(f"{mode}_aux_loss", loss1, prog_bar=False, sync_dist=True)

        loss = embedding_loss + 0.2 * loss1
        self.log(f"{mode}_loss", loss, prog_bar=False, sync_dist=True)

        # Calaculate accuracy
        preds = torch.argmax(audio_aux, dim=-1)
        acc = (preds[mask] == phones[mask]).float().mean()

        self.log(f"{mode}_acc", acc, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=2e-5, weight_decay=1e-2, betas=(0.9, 0.98)
        )

        return optimizer


if __name__ == "__main__":
    pl.seed_everything(42)

    train_dataset = WhisperDataset(split="train")
    val_dataset = WhisperDataset(split="val")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = WhisperModel()

    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=1,
        accumulate_grad_batches=16,
        max_epochs=100,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}-{val_acc:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        ],
        logger=WandbLogger(
            project="whisper",
            save_dir="logs",
            entity="fish-audio",
        ),
        resume_from_checkpoint="logs/whisper/d2mevxji/checkpoints/epoch=50-val_loss=0.0789-val_acc=0.8715.ckpt",
    )

    trainer.fit(model, train_loader, val_loader)
