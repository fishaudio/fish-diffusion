from torch.utils.data import DataLoader

from .builder import DATASETS
from .repeat import RepeatDataset


def build_loader_from_config(cfg, num_devices=1):
    train_dataset = DATASETS.build(cfg.dataset.train)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = DATASETS.build(cfg.dataset.valid)

    if num_devices > 1:
        valid_dataset = RepeatDataset(
            valid_dataset, repeat=num_devices, collate_fn=valid_dataset.collate_fn
        )

    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    return train_loader, valid_loader
