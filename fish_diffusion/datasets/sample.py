import random
from typing import Union

from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class SampleDataset(Dataset):
    def __init__(
        self, dataset: Union[dict, Dataset], num_samples: int = 8, collate_fn=None
    ) -> None:
        """Sample a dataset. Useful for DDP training.

        Args:
            dataset (Union[dict, Dataset]): Dataset to sample.
            num_samples (int): Number of samples to sample.
            collate_fn (Callable, optional): Collate function. Defaults to None.
        """

        super().__init__()

        self.num_samples = num_samples
        self.collate_fn = collate_fn

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.dataset) - 1)
        return self.dataset[idx]
