import zipfile
from io import BytesIO
from pathlib import Path

import click
import requests
from loguru import logger
from tqdm import tqdm

DOWNLOAD_URL = "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip"


@click.command()
@click.option(
    "--target_dir",
    "-t",
    default="checkpoints",
    help="Directory to save the model.",
)
@click.option(
    "--use-ghproxy",
    default=False,
    help="Use ghproxy.com for speed up",
    is_flag=True,
)
@click.option(
    "--agree-license",
    default=False,
    help="You argee the CC BY-NC-SA 4.0 license.",
    is_flag=True,
)
def main(
    target_dir: str = "checkpoints",
    use_ghproxy: bool = False,
    agree_license: bool = False,
):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Show license Information
    logger.info("NSF HifiGan (OpenVPI) is released under the CC BY-NC-SA 4.0 license.")
    logger.info(
        "See https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1 for more information."
    )

    if agree_license:
        logger.info("You already argee the license by passing --agree-license option.")
    else:
        # Check if user agrees to the license
        agree = input("Do you agree to the license? [y/N] ")
        if agree.lower() != "y":
            logger.error("You must agree to the license to download this model.")
            return

    # Download the model
    logger.info("Downloading the model...")

    url = DOWNLOAD_URL
    if use_ghproxy:
        url = f"https://ghproxy.com/{DOWNLOAD_URL}"
        logger.info("Using ghproxy.com for speed up downloading.")

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading")

    f = BytesIO()
    for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)

    t.close()

    # Unzip the model
    logger.info("Unzipping the model...")
    f.seek(0)

    with zipfile.ZipFile(f, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
