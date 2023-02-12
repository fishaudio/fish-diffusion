from pathlib import Path

import librosa
import soundfile as sf
from fish_audio_preprocess.utils.file import list_files
from tqdm import tqdm

path = Path("dataset/mfa-data/english/LibriSpeech")

for file in list_files(path, ".lab", recursive=True):
    file.unlink()

for transcript in tqdm(list_files(path, ".trans.txt", recursive=True)):
    for i in transcript.read_text().splitlines():
        file, text = i.strip().split(" ", 1)
        file = transcript.parent / file

        audio = file.with_suffix(".wav")
        if not audio.exists():
            audio = file.with_suffix(".flac")

            if not audio.exists():
                raise ValueError(f"Cannot find audio file for {file}")

            audio, sr = librosa.load(str(audio), sr=None, mono=True)
            sf.write(str(file.with_suffix(".wav")), audio, sr)

        file.with_suffix(".lab").write_text(text)
