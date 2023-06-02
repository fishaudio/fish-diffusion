from typing import Iterable

from torch.utils.data import ConcatDataset as _ConcatDataset

# from .builder import DATASETS
from hydra.utils import get_static_method
from loguru import logger


# @DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets: Iterable[dict], collate_fn=None) -> None:
        """Concatenate multiple datasets.

        Args:
            datasets (Iterable[dict]): Datasets to concatenate.
            collate_fn (Callable, optional): Collate function. Defaults to None.
        """

        # super().__init__([DATASETS.build(dataset) for dataset in datasets])
        super().__init__([dataset for dataset in datasets])

        if isinstance(collate_fn, str):
            self.collate_fn = get_static_method(collate_fn)
        else:
            self.collate_fn = collate_fn
