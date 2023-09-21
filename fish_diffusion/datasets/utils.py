from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .builder import DATASETS
from .repeat import RepeatDataset
from .sample import SampleDataset


def build_loader_from_config(cfg, num_devices=1):
    train_dataset = DATASETS.build(cfg.dataset.train)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        **cfg.dataloader.train,
    )

    valid_dataset = DATASETS.build(cfg.dataset.valid)

    if num_devices > 1 and isinstance(valid_dataset, (RepeatDataset, SampleDataset)):
        valid_dataset = RepeatDataset(
            valid_dataset, repeat=num_devices, collate_fn=valid_dataset.collate_fn
        )

    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        **cfg.dataloader.valid,
    )

    return train_loader, valid_loader


def pad_and_stack(x, dim=0):
    if isinstance(x[0], np.ndarray):
        x = [torch.from_numpy(i).float() for i in x]

    lens = torch.LongTensor([i.shape[dim] for i in x])
    max_len = torch.max(lens)

    if dim < 0:
        pads = (0,) * (abs(dim + 1) * 2)
    else:
        negative_pad_dim = dim - len(x[0].shape) + 1
        pads = (0,) * (abs(negative_pad_dim) * 2)

    stacked = torch.stack(
        [torch.nn.functional.pad(i, pads + (0, max_len - i.shape[dim])) for i in x]
    )

    return (
        stacked,
        lens,
        max_len,
    )


def get_speaker_map_from_subfolder(path, existing_speaker_map=None):
    if existing_speaker_map is None:
        speaker_map = {}
    else:
        speaker_map = deepcopy(existing_speaker_map)

    for speaker_path in sorted(Path(path).iterdir()):
        if not speaker_path.is_dir() or speaker_path.name.startswith("."):
            continue

        speaker_map[str(speaker_path.name)] = len(speaker_map)

    return speaker_map


def get_datasets_from_subfolder(
    type, path, speaker_map: dict[str, int], *args, **kwargs
) -> list[dict]:
    datasets = []

    for speaker_path in sorted(Path(path).iterdir()):
        if not speaker_path.is_dir() or speaker_path.name.startswith("."):
            continue

        speaker_id = speaker_map[str(speaker_path.name)]
        datasets.append(
            dict(
                type=type,
                path=str(speaker_path),
                speaker_id=speaker_id,
                *args,
                **kwargs,
            )
        )

    return datasets


def transform_pipeline(pipeline, data):
    for step in pipeline:
        if step["type"] == "PickKeys":
            new_data = {}
            for k in step["keys"]:
                if isinstance(k, tuple):
                    new_data[k[0]] = data[k[1]]
                else:
                    new_data[k] = data[k]
            data = new_data
        elif step["type"] == "ListToDict":
            all_keys = (
                set(j for i in data for j in i.keys())
                if "keys" not in step
                else step["keys"]
            )
            data = {k: [i[k] for i in data] for k in all_keys}
        elif step["type"] == "PadStack":
            for k, v in step["keys"]:
                stacked, lens, max_len = pad_and_stack(data[k], v)
                data[k] = stacked
                data[k + "_lens"] = lens
                data[k + "_max_len"] = max_len
        elif step["type"] == "ToTensor":
            for k, t in step["keys"]:
                if isinstance(data[k], np.ndarray):
                    data[k] = torch.from_numpy(data[k]).type(t)
                elif isinstance(data[k], torch.Tensor):
                    data[k] = data[k].type(t)
                else:
                    data[k] = torch.tensor(data[k], dtype=t)
        elif step["type"] == "Transpose":
            for k, *args in step["keys"]:
                data[k] = data[k].transpose(*args)
        elif step["type"] == "UnSqueeze":
            for k, *args in step["keys"]:
                if isinstance(data[k], np.ndarray):
                    data[k] = np.expand_dims(data[k], *args)
                else:
                    data[k] = data[k].unsqueeze(*args)
        elif step["type"] == "FilterByLength":
            data = [
                i
                for i in data
                if step["min_length"]
                <= i[step["key"]].shape[step["dim"]]
                <= step["max_length"]
            ]
        else:
            raise NotImplementedError(f"Unknown transform type: {step['type']}")

    return data
