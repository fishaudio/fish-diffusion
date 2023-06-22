from typing import Iterable, Union

from hydra.utils import instantiate
from loguru import logger
from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    def __init__(
        self, dataset: Union[dict, Dataset], repeat: int, collate_fn=None
    ) -> None:
        """Repeat a dataset. Useful for DDP training.

        Args:
            dataset (Union[dict, Dataset]): Dataset to repeat.
            repeat (int): Number of times to repeat.
            collate_fn (Callable, optional): Collate function. Defaults to None.
        """

        self.repeat = repeat
        self.collate_fn = collate_fn

        if isinstance(dataset, dict):
            self.dataset = instantiate(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx // self.repeat]
