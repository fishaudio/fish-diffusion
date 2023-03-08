import random
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm


@click.command()
@click.argument(
    "input",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("num", required=True, type=int)
@click.option("--seed", default=42, help="Random seed", type=int)
def main(input: Path, output: Path, num: int, seed: int):
    """
    Randomly move NUM files from INPUT to OUTPUT.
    """

    random.seed(seed)
    logger.info(f"Random seed: {seed}")
    logger.info(f"Input folder: {input}")
    logger.info(f"Output folder: {output}")
    logger.info(f"Number of random moves: {num}")

    all_files = list([path for path in input.iterdir() if path.is_file()])
    logger.info(f"Movable files: {len(all_files)}")

    if num > len(all_files):
        logger.error(
            f"Number of random moves ({num}) is greater than the number of movable files ({len(all_files)})"
        )
        return

    random.shuffle(all_files)

    if output.exists() is False:
        output.mkdir(parents=True)

    for i in tqdm(range(num)):
        path = all_files[i]
        new_path = output / path.name
        path.rename(new_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
