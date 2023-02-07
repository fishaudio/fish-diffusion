from pathlib import Path

import librosa
import soundfile as sf
from fish_audio_preprocess.utils.file import list_files
from tqdm import tqdm

path = Path("dataset/mfa-data/english/LJSpeech")

for file in list_files(path, ".lab", recursive=True):
    file.unlink()

metadata = path / "metadata.csv"
metadata = metadata.read_text().splitlines()

for i in tqdm(metadata):
    file, _, text = i.split("|")

    # Normalize text, remove punctuation
    text = text.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    text = text.lower()

    file = path / "wavs" / file
    file.with_suffix(".lab").write_text(text)
