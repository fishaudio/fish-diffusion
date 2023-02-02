from pathlib import Path

from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from pypinyin import Style, pinyin

# Clean all .lab files in a directory
path = Path("dataset/mfa-data/opencpop/segments")
for file in list_files(path, ".lab", recursive=True):
    file.unlink()

logger.info("All .lab files have been deleted")


trascription = Path("dataset/mfa-data/opencpop/transcriptions.txt")
for i in trascription.open():
    id, text, *_ = i.split("|")
    p = [i[0] for i in pinyin(text, style=Style.NORMAL, strict=True)]
    text = " ".join(p)

    file = path / (id + ".lab")
    file.write_text(text)

logger.info("All .txt files have been converted to .lab files")
