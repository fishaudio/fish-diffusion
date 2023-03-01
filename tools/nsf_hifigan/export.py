import click
import torch
from loguru import logger


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument(
    "output_file", type=click.Path(), required=False, default="nsf_hifigan.pt"
)
def main(input_file, output_file):
    logger.info(f"Exporting {input_file}")

    checkpoint = torch.load(input_file, map_location="cpu")
    model = checkpoint["state_dict"]

    generator_params = {
        k.replace("generator.", ""): v
        for k, v in model.items()
        if k.startswith("generator.")
    }

    torch.save(
        {
            "generator": generator_params,
        },
        output_file,
    )

    logger.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
