import zipfile
from io import BytesIO
from pathlib import Path

import sys

import requests
from loguru import logger
from tqdm import tqdm

DOWNLOAD_URL = "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip"

try:
    if(sys.argv.index("--use-ghproxy")):
        logger.info("Using Ghproxy.com for speed up")
        DOWNLOAD_URL = "https://ghproxy.com/" + DOWNLOAD_URL
except:
    pass

AGREE_LICENSE = False

try:
    if(sys.argv.index("-y")):
        logger.info(
            "You already argee the license.See https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1 for more information."
        )
        AGREE_LICENSE = True
except:
    pass

def main(target_dir: str = "checkpoints"):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Show license Information
    if(!AGREE_LICENSE):
        logger.info("NSF HifiGan (OpenVPI) is released under the CC BY-NC-SA 4.0 license.")
        logger.info(
            "See https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1 for more information."
        )

        # Check if user agrees to the license
        agree = input("Do you agree to the license? [y/N] ")
        if agree.lower() != "y":
            logger.error("You must agree to the license to download this model.")
            return

    # Download the model
    logger.info("Downloading the model...")
    r = requests.get(DOWNLOAD_URL, stream=True)
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
