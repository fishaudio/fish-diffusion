import click
import torch
from loguru import logger


@click.command()
@click.option(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
)
@click.option("--output", type=str, required=True, help="Path to the output audio file")
@click.option(
    "--remove-speaker-embeddings",
    is_flag=True,
    help="Remove speaker embeddings",
    default=False,
)
@click.option(
    "--remove-optimizer-states",
    is_flag=True,
    help="Remove optimizer states",
    default=False,
)
@click.option(
    "--remove-discriminator", is_flag=True, help="Remove discriminator", default=False
)
@click.option(
    "--remove-generator", is_flag=True, help="Remove generator", default=False
)
def main(
    checkpoint: str,
    output: str,
    remove_speaker_embeddings: bool,
    remove_optimizer_states: bool,
    remove_discriminator: bool,
    remove_generator: bool,
):
    data = torch.load(checkpoint, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint}")

    if remove_optimizer_states and "state_dict" in data:
        data = data["state_dict"]

    if remove_speaker_embeddings:
        del data["generator.speaker_encoder.embedding.weight"]

    if remove_discriminator:
        data = {
            k: v
            for k, v in data.items()
            if not k.startswith("mpd") and not k.startswith("msd")
        }

    if remove_generator:
        data = {k: v for k, v in data.items() if not k.startswith("generator")}

    torch.save(data, output)
    logger.info(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
