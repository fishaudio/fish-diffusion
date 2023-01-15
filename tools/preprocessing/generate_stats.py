import json

import numpy as np
from fish_audio_preprocess.utils.file import list_files
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)

    return parser.parse_args()


def process(args):
    mel_min = np.ones(128) * float("inf")
    mel_max = np.ones(128) * -float("inf")

    for wav_path in list_files(args.input_dir, {".wav"}, recursive=True, sort=True):
        mel_spectrogram_path = wav_path.parent / f"{wav_path.name}.mel.npy"
        mel_spectrogram = np.load(mel_spectrogram_path)

        m_min = np.min(mel_spectrogram, axis=1)
        m_max = np.max(mel_spectrogram, axis=1)

        mel_min = np.minimum(mel_min, m_min)
        mel_max = np.maximum(mel_max, m_max)

    with open(args.output_file, "w") as f:
        stats = {
            "spec_min": mel_min.tolist(),
            "spec_max": mel_max.tolist(),
        }
        f.write(json.dumps(stats))


if __name__ == "__main__":
    args = parse_args()

    process(args)
