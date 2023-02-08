from pathlib import Path

from fish_audio_preprocess.utils.file import list_files
from tqdm import tqdm

from_path = Path("dataset/mfa-data/english/LJSpeech/labeled")
to_path = Path("dataset/mfa-data/english/LJSpeech/normed")

for file in tqdm(list_files(from_path, ".TextGrid", recursive=True)):
    data = file.read_bytes()
    new_path = to_path / file.relative_to(from_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(data)
