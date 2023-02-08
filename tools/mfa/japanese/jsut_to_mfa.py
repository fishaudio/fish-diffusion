from pathlib import Path

import pykakasi
from fish_audio_preprocess.utils.file import list_files

kks = pykakasi.kakasi()

path = Path("dataset/mfa-data/japanese/jsut_ver1.1")

for file in list_files(path, ".lab", recursive=True):
    file.unlink()

for subset in path.iterdir():
    if not subset.is_dir():
        continue

    transcript = subset / "transcript_utf8.txt"

    with transcript.open("r", encoding="utf-8") as f:
        lines = [i.strip() for i in f.readlines()]

    for line in lines:
        file, text = line.split(":")
        file = subset / "wav" / file

        # Normalize text
        text = "".join([i for i in text if i not in ["，", "、", "。"]])
        text = " ".join([i["kana"] for i in kks.convert(text)])

        file.with_suffix(".lab").write_text(text)
