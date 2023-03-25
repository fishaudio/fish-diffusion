import zipfile
from io import BytesIO
from pathlib import Path

import click
import requests
from loguru import logger
from tqdm import tqdm

DOWNLOAD_URLS = {
    "OpenVPI": "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip",
    "FishAudioBeta": "https://github.com/fishaudio/fish-diffusion/releases/download/v1.12/nsf_hifigan-beta-v2-epoch-434.zip",
    "FishAudioStableV1": "https://github.com/fishaudio/fish-diffusion/releases/download/v2.0.0/nsf_hifigan-stable-v1.zip",
}


def download_model(file, model: str, use_ghproxy: bool = False):
    url = DOWNLOAD_URLS[model]
    if use_ghproxy:
        url = f"https://ghproxy.com/{url}"
        logger.info("Using ghproxy.com for speed up downloading.")

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading")

    for data in r.iter_content(block_size):
        t.update(len(data))
        file.write(data)


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
@click.option(
    "--vocoder",
    default="FishAudioStableV1",
    help="Model to download",
    type=click.Choice(["OpenVPI", "FishAudioBeta", "FishAudioStableV1"]),
)
def main(
    target_dir: str = "checkpoints",
    use_ghproxy: bool = False,
    agree_license: bool = False,
    vocoder: str = "FishAudioStableV1",
):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Show license Information
    logger.info(
        f"NSF HifiGan ({vocoder}) is released under the CC BY-NC-SA 4.0 license."
    )
    logger.info(
        "See https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1 for more information."
        if vocoder == "OpenVPI"
        else "See https://github.com/fishaudio/fish-diffusion/releases for more information."
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
    logger.info("Downloading the Vocoder...")
    f = BytesIO()
    download_model(f, vocoder, use_ghproxy)
    f.seek(0)

    # Unzip the model
    logger.info("Unzipping the Vocoder...")

    with zipfile.ZipFile(f, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
