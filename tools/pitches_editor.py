from pathlib import Path

import click
import librosa
import matplotlib.dates as mdates
import numpy as np
import torch
from librosa.core import hz_to_mel, mel_frequencies
from loguru import logger
from matplotlib import pyplot as plt

from fish_diffusion.modules.pitch_extractors import (
    CrepePitchExtractor,
    DioPitchExtractor,
    HarvestPitchExtractor,
    ParselMouthPitchExtractor,
)
from fish_diffusion.utils.audio import get_mel_from_audio

# Define global variables
workspace = Path("pitches_editor")
if not workspace.exists():
    workspace.mkdir(parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

f_min = 40
f_max = 16000
n_mels = 128

min_mel = hz_to_mel(f_min)
max_mel = hz_to_mel(f_max)
f_to_mel = lambda x: (hz_to_mel(x) - min_mel) / (max_mel - min_mel) * n_mels
mel_freqs = mel_frequencies(n_mels=n_mels, fmin=f_min, fmax=f_max)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("path")
def extract(path):
    audio, sr = librosa.load(
        path,
        sr=44100,
        mono=True,
    )
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    mel = (
        get_mel_from_audio(audio, sr, f_min=f_min, f_max=f_max, n_mels=n_mels)
        .cpu()
        .numpy()
    )
    logger.info(f"Got mel spectrogram with shape {mel.shape}")
    np.save(workspace / "mel.npy", mel)

    extractors = {
        "Crepe": CrepePitchExtractor,
        "ParselMouth": ParselMouthPitchExtractor,
        "Dio": DioPitchExtractor,
    }

    pitches = {}

    for name, extractor in extractors.items():
        pitch_extractor = extractor(f0_min=40.0, f0_max=1600, keep_zeros=False)
        f0 = pitch_extractor(audio, sr, pad_to=mel.shape[-1]).cpu().numpy()
        logger.info(f"Got {name} pitch with shape {f0.shape}")

        np.save(workspace / f"{name}.npy", f0)
        pitches[name] = f0.tolist()

    pitches["final"] = pitches["Crepe"]
    data = {
        "mel": mel.tolist(),
        "pitches": pitches,
    }

    import json

    with open(workspace / "data.json", "w") as f:
        json.dump(data, f)


@cli.command()
def plot():
    mel = np.load(workspace / "mel.npy")

    all_pitches = {
        k.stem: np.load(k)
        for k in workspace.iterdir()
        if k.suffix == ".npy" and k.stem != "mel"
    }

    _, axs = plt.subplots(
        len(all_pitches),
        1,
        figsize=(10, len(all_pitches) * 3),
        sharex=True,
        sharey=True,
    )

    for idx, (name, f0) in enumerate(all_pitches.items()):
        f0 = f_to_mel(f0)
        f0[f0 <= 0] = float("nan")

        ax = axs[idx]
        ax.set_title(name)

        ax.imshow(mel, aspect="auto", origin="lower")
        ax.plot(f0, label=name, color="red")
        ax.set_yticks(np.arange(0, 128, 10))
        ax.set_yticklabels(np.round(mel_freqs[::10]).astype(int))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.legend()

    plt.savefig("pitch.png")
    plt.show()


@cli.command()
@click.argument("input_name")
@click.argument("output_name")
@click.argument("begin", type=int)
@click.argument("end", type=int)
def apply(input_name, output_name, begin, end):
    input = np.load(workspace / f"{input_name}.npy")
    output = np.load(workspace / f"{output_name}.npy")

    output[begin:end] = input[begin:end]

    np.save(workspace / f"{output_name}.npy", output)
    logger.info(f"Applied {input_name} to {output_name}")


if __name__ == "__main__":
    cli()
