import shutil
import urllib
from pathlib import Path

import click
import torch
from loguru import logger

from fish_diffusion.modules.vocoders.nsf_hifigan.nsf_hifigan import NsfHifiGAN


class ExportableNsfHiFiGAN(NsfHifiGAN):
    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        mel = mel.transpose(2, 1) * 2.30259
        wav = self.model(mel, f0)

        return wav


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument(
    "output_path", type=click.Path(), required=False, default="exported/nsf_hifigan"
)
@click.option(
    "--config", type=str, required=False, default="tools/nsf_hifigan/config_v1.json"
)
@click.option(
    "--license",
    type=str,
    required=False,
    default="https://raw.githubusercontent.com/vitorsr/cc/master/CC-BY-NC-SA-4.0.md",
)
def main(input_file, output_path, config, license):
    output_path = Path(output_path)
    if output_path.exists():
        logger.warning(f"Output path {output_path} already exists, deleting")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting {input_file} to {output_path}")

    checkpoint = torch.load(input_file, map_location="cpu")
    model = checkpoint["state_dict"]

    generator_params = {
        k.replace("generator.", ""): v
        for k, v in model.items()
        if k.startswith("generator.")
    }

    pt_path = output_path / "model"
    torch.save(
        {
            "generator": generator_params,
        },
        pt_path,
    )

    logger.info(f"Exported to {pt_path}, now exporting config")

    shutil.copy(config, output_path / "config.json")
    logger.info(f"Config exported")

    # Export ONNX
    logger.info(f"Exporting ONNX")
    model = ExportableNsfHiFiGAN(checkpoint_path=input_file, config_file=config)
    model.eval()
    logger.info(f"Model loaded")

    mel = torch.randn(1, 80, 128)
    f0 = torch.randn(1, 80)

    torch.onnx.export(
        model,
        (mel, f0),
        output_path / "nsf_hifigan.onnx",
        input_names=["mel", "f0"],
        output_names=["waveform"],
        opset_version=16,
        dynamic_axes={
            "mel": {0: "batch", 1: "n_frames"},
            "f0": {0: "batch", 1: "n_frames"},
            "waveform": {0: "batch", 2: "wave_length"},
        },
    )

    logger.info(f"ONNX exported")

    # Export license
    if license:
        logger.info(f"Exporting license")
        if license.startswith("http"):
            urllib.request.urlretrieve(license, output_path / "LICENSE")
        else:
            shutil.copy(license, output_path / "LICENSE")
        logger.info(f"License exported")

    logger.info(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
