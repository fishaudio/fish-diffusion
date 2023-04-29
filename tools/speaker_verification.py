"""
This script is under development. It will be used to evaluate the timbre leaking problem.
"""

from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Inference, Model
from tqdm import tqdm

model = Model.from_pretrained("pyannote/embedding")

inference = Inference(model, window="whole")

speakers = ["CDF1", "CDM1", "IDF1", "IDM1"]
train_embeddings = []

for speaker in tqdm(speakers, desc="Train Split"):
    temp = []
    for i in Path(f"dataset/svcc/train/{speaker}").glob("*.wav"):
        temp.append(inference(str(i)))

    train_embeddings.append(np.mean(temp, axis=0))

valid_embeddings = []
valid_subset = ["SF1", "SM1"]

for speaker in tqdm(speakers, desc="Valid Split"):
    temp = []
    for subset in valid_subset:
        for i in Path(
            f"results/SVCC2023DatasetEvaluation/Fish/{subset}/{speaker}"
        ).glob("*.wav"):
            temp.append(inference(str(i)))

    valid_embeddings.append(np.mean(temp, axis=0))

# embedding1 = inference("results/SVCC2023DatasetEvaluation/Fish/SF1/IDF1/30001.wav")
# embedding2 = inference("dataset/svcc/train/IDF1/10001.wav")
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

# print(embedding1.shape)
# print(embedding2.shape)

from scipy.spatial.distance import cdist

distance = cdist(valid_embeddings, train_embeddings, metric="cosine")

# Plot the confusion matrix
import matplotlib.pyplot as plt

# yellow color scheme
plt.imshow(distance)
# Add text
for i in range(len(speakers)):
    for j in range(len(speakers)):
        plt.text(i, j, round(distance[i, j], 2), ha="center", va="center", color="w")

plt.colorbar()
plt.xticks(range(len(speakers)), speakers)
plt.yticks(range(len(speakers)), speakers)
plt.xlabel("Train")
plt.ylabel("Valid")
plt.title("Cosine Distance")
plt.tight_layout()

plt.savefig("distance.png")

print(distance)
