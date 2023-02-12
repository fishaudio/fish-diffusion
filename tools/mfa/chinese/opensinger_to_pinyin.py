from pathlib import Path

from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from pypinyin import Style, pinyin

# Clean all .lab files in a directory
path = Path("dataset/mfa-data/OpenSinger")
for file in list_files(path, ".lab", recursive=True):
    file.unlink()

logger.info("All .lab files have been deleted")

# Convert all .txt files to .lab files
for file in list_files(path, ".txt", recursive=True):
    p = [i[0] for i in pinyin(file.read_text(), style=Style.NORMAL, strict=True)]
    text = " ".join(p)

    file.with_suffix(".lab").write_text(text)

logger.info("All .txt files have been converted to .lab files")
