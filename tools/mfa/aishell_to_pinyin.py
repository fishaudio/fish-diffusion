from pathlib import Path

from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from pypinyin import Style, pinyin

# Clean all .lab files in a directory
path = Path("dataset/mfa-data/aishell")
for file in list_files(path, ".lab", recursive=True):
    file.unlink()

logger.info("All .lab files have been deleted")

# Convert all .txt files to .lab files
for line in Path("dataset/mfa-data/aishell/content.txt").open():
    file, text = line.strip().split("\t")

    text = [
        i
        for idx, i in enumerate(text.split(" "))
        if i not in ["", "%", "$"] and idx % 2 == 0
    ]

    text = "".join(text)
    text = [i[0] for i in pinyin(text, style=Style.NORMAL, strict=True)]

    text = " ".join(text)

    file = path / "wav" / file[:7] / file.replace(".wav", ".lab")
    file.write_text(text)

logger.info("All .txt files have been converted to .lab files")
