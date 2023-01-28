from typing import Iterable

from torch.utils.data import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets: Iterable[dict], collate_fn=None) -> None:
        """Concatenate multiple datasets.

        Args:
            datasets (Iterable[dict]): Datasets to concatenate.
            collate_fn (Callable, optional): Collate function. Defaults to None.
        """

        super().__init__([DATASETS.build(dataset) for dataset in datasets])

        self.collate_fn = collate_fn
