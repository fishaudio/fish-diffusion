from pathlib import Path

from fish_audio_preprocess.utils.file import list_files
from loguru import logger
from montreal_forced_aligner import align
from pypinyin import Style, pinyin
from textgrid import TextGrid

# Clean all .lab files in a directory
path = Path("dataset/mfa-data/M4Singer")
for file in list_files(path, ".lab", recursive=True):
    file.unlink()

logger.info("All .lab files have been deleted")


# Convert all .txt files to .lab files
for file in list_files(path, ".TextGrid", recursive=True):
    # p = [i[0] for i in pinyin(file.read_text(), style=Style.NORMAL, strict=True)]
    # text = " ".join(p)

    grid = TextGrid.fromFile(str(file))
    text = [i.mark for i in grid.tiers[0] if i.mark not in ["<AP>", "<SP>", ""]]
    text = "".join(text)
    text = [i[0] for i in pinyin(text, style=Style.NORMAL, strict=True)]
    text = " ".join(text)

    file.with_suffix(".lab").write_text(text)
    file.rename(file.with_suffix(".TextGrid.bak"))

logger.info("All .txt files have been converted to .lab files")
