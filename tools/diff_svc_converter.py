from argparse import ArgumentParser

import torch
from mmengine import Config

from train import FishDiffusion

KEYS_MAPPING = {
    "betas": "diffusion.betas",
}


def convert(config_path, input_path, output_path):
    config = Config.fromfile(config_path)
    model = FishDiffusion(config)

    diff_svc_state_dict = torch.load(
        input_path,
        map_location="cpu",
    )["state_dict"]
    # diff_svc_state_dict_clean = {k[6:]: v for k, v in diff_svc_state_dict.items() if k.startswith("model.") and "fs2" not in k}

    diffusion_state_dict = {}
    denoiser_state_dict = {}
    fs2_state_dict = (
        {}
    )  # FS2 is not used in this model, only projection layers are keeped.

    for k, v in diff_svc_state_dict.items():
        if not k.startswith("model."):
            raise ValueError(f"Key {k} does not start with model.")

        k = k[6:]

        if k.startswith("fs2."):
            fs2_state_dict[k[4:]] = v
            continue

        if k.startswith("denoise_fn."):
            denoiser_state_dict[k[11:]] = v
            continue

        assert "." not in k, f"Key {k} contains dot, which is not supported yet."

        diffusion_state_dict[k] = v

    fish_denoiser_keys = list(model.model.diffusion.denoise_fn.state_dict().keys())

    for i in fish_denoiser_keys:
        fixed = (
            i.replace(".conv.", ".")
            .replace(".linear.", ".")
            .replace(".conv_layer.", ".dilated_conv.")
        )
        diffusion_state_dict[f"denoise_fn.{i}"] = denoiser_state_dict.pop(fixed)

    assert (
        len(denoiser_state_dict) == 0
    ), f"Unmapped denoiser keys: {denoiser_state_dict.keys()}"
    model.model.diffusion.load_state_dict(diffusion_state_dict, strict=True)

    # The Main diffusion is done now

    # Fix Pitch Encoder
    model.model.pitch_encoder.embedding.weight.data = fs2_state_dict[
        "pitch_embed.weight"
    ]

    # TODO: Support multi speaker
    model.model.speaker_encoder.embedding.weight.data.zero_()

    torch.save(model.state_dict(), output_path)

    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/svc_hubert_soft_diff_svc.py"
    )
    parser.add_argument(
        "--input-path", type=str, required=True, help="The input checkpoint (Diff-SVC)"
    )
    parser.add_argument(
        "--output-path", type=str, help="The output checkpoint (Fish-Diffusion)"
    )
    args = parser.parse_args()

    convert(args.config, args.input_path, args.output_path)
