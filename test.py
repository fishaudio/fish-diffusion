import numpy as np
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("bert-base-cased")
d = np.load(
    "dataset/LibriTTS/train-clean-100/26/495/26_495_000004_000000.0.data.npy",
    allow_pickle=True,
).item()
print(t.decode(d["contents"][0]))
