from pathlib import Path
from typing import Union


def load_dictionary(
    path: Union[str, Path], with_special_tokens: bool = True
) -> tuple[dict[str, list[str]], list[str]]:
    """Load dictionary from file

    Args:
        path (Union[str, Path]): path to dictionary file
        with_special_tokens (bool, optional): add AP and SP tokens. Defaults to True.

    Returns:
        dict[str, list[str]]: dictionary
        list[str]: unique phones
    """

    pinyin_to_phones = {}

    with open(path, "r") as f:
        for line in f:
            pinyin, phones = line.strip().split("\t")
            pinyin_to_phones[pinyin] = phones.split(" ")

    unique_phones = set(
        [phone for phones in pinyin_to_phones.values() for phone in phones]
    )
    unique_phones = sorted(list(unique_phones))

    if with_special_tokens:
        unique_phones = ["AP", "SP"] + unique_phones

    return pinyin_to_phones, unique_phones
