from pathlib import Path

import autofaiss
import librosa
import numpy as np
import torch
from tqdm import tqdm

source_path = Path("dataset/train")
for speaker_path in tqdm(list(source_path.iterdir()), desc="Speaker", position=0):
    if speaker_path.is_file():
        continue

    speaker = speaker_path.name
    contents = []

    for i in tqdm(list(speaker_path.rglob("*.npy")), desc="Content", position=1):
        data = np.load(i, allow_pickle=True)
        contents.append(data.item()["contents"])

    contents = np.concatenate(contents, axis=1).T

    autofaiss.build_index(
        contents,
        str(source_path / f"{speaker}.index"),
        str(source_path / f"{speaker}.index.json"),
        metric_type="ip",
    )
