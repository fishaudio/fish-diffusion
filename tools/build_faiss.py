from pathlib import Path

import autofaiss
import numpy as np
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
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)
    print(f"Buiding index for {speaker}..., shape: {contents.shape}")

    autofaiss.build_index(
        contents,
        str(source_path / f"{speaker}.index"),
        str(source_path / f"{speaker}.index.json"),
        # HNSW will use more memory than IVF, but it's faster
        index_key="IVF1024,Flat",
        # index_key="HNSW64",
        metric_type="ip",  # inner product, we use cosine similarity
    )
