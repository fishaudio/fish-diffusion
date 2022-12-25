import json
import os

import numpy as np

max_seq_len = -float('inf')
mel_min = np.ones(128) * float('inf')
mel_max = np.ones(128) * -float('inf')


for wavpath in open("filelists/train.txt").readlines():
    wavpath = wavpath.strip()
    mel_spectrogram = np.load(wavpath+".mel.npy")
    m_min = np.min(mel_spectrogram, axis=1)
    m_max = np.max(mel_spectrogram, axis=1)

    mel_min = np.minimum(mel_min, m_min)
    mel_max = np.maximum(mel_max, m_max)
with open(os.path.join("dataset", "stats.json"), "w") as f:
    stats = {
        "spec_min": mel_min.tolist(),
        "spec_max": mel_max.tolist(),
    }
    f.write(json.dumps(stats))

