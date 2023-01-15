from torch.utils.data import ConcatDataset as _ConcatDataset
from .builder import DATASETS
from typing import Iterable


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets: Iterable[dict]) -> None:
        """Concatenate multiple datasets.

        Args:
            datasets (Iterable[dict]): Datasets to concatenate.
        """

        super().__init__([DATASETS.build(dataset) for dataset in datasets])
