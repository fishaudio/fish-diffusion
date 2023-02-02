from argparse import ArgumentParser

import torch
from loguru import logger
from mmengine import Config

from train import FishDiffusion


def convert(config_path, input_path, output_path):
    config = Config.fromfile(config_path)
    model = FishDiffusion(config)

    logger.info("Loading Diff-SVC checkpoint...")

    diff_svc_state_dict = torch.load(
        input_path,
        map_location="cpu",
    )["state_dict"]

    # Detect if residual channels mismatch
    residual_channels = diff_svc_state_dict[
        "model.denoise_fn.input_projection.weight"
    ].shape[0]
    if residual_channels != config.model.diffusion.denoiser.residual_channels:
        logger.error(
            f"Residual channels mismatch: {residual_channels} vs {config.model.diffusion.denoiser.residual_channels}. "
            f"Please update the `model.diffusion.denoiser.residual_channels` to {residual_channels} in the config file."
        )
        return

    logger.info(f"Residual channels: {residual_channels}")

    # Detect if spec_min and spec_max mismatch
    spec_min = diff_svc_state_dict["model.spec_min"].shape[-1]
    spec_max = diff_svc_state_dict["model.spec_max"].shape[-1]
    config_spec_min = model.model.diffusion.spec_min.shape[-1]

    if not spec_min == spec_max == config_spec_min:
        logger.error(
            f"Spec min and max shape mismatch: {spec_min} vs {spec_max} vs {config_spec_min}. "
            f"Please update the `model.diffusion.spec_min` and `model.diffusion.spec_max` to [0] * {spec_min} in the config file."
        )
        return

    logger.info(f"Spec min and max shape: {spec_min}")

    # Solving diffusion and denoisr params
    fish_denoiser_keys = list(model.model.diffusion.state_dict().keys())
    diffusion_state_dict = {}

    for i in fish_denoiser_keys:
        fixed = "model." + (
            i.replace(".conv.", ".")
            .replace(".linear.", ".")
            .replace(".conv_layer.", ".dilated_conv.")
        )
        diffusion_state_dict[i] = diff_svc_state_dict.pop(fixed)

    # If any keys not beginning with "model.fs2" are left, they are not mapped
    if any(not k.startswith("model.fs2") for k in diff_svc_state_dict.keys()):
        logger.error(f"Keys not mapped: {diff_svc_state_dict.keys()}")
        return

    model.model.diffusion.load_state_dict(diffusion_state_dict, strict=True)
    logger.info("Diffusion and Denoiser are converted.")

    # Restoring Pitch Encoder
    pitch_encoder_state_dict = {
        "embedding.weight": diff_svc_state_dict.pop("model.fs2.pitch_embed.weight"),
    }
    model.model.pitch_encoder.load_state_dict(pitch_encoder_state_dict, strict=True)
    logger.info("Pitch Encoder is converted.")

    # Restoring Speaker Encoder
    if "model.fs2.spk_embed_proj.weight" in diff_svc_state_dict:
        speaker_encoder_state_dict = {
            "embedding.weight": diff_svc_state_dict.pop(
                "model.fs2.spk_embed_proj.weight"
            ),
        }

        num_speakers = model.model.speaker_encoder.embedding.weight.shape[0]
        diff_svc_num_speakers = speaker_encoder_state_dict["embedding.weight"].shape[0]

        if diff_svc_num_speakers != num_speakers:
            logger.error(
                f"Speaker number mismatch: {diff_svc_num_speakers} vs {num_speakers}. "
                f"Please update the speaker_encoder.input_size to {diff_svc_num_speakers} in the config file."
            )
            return

        model.model.speaker_encoder.load_state_dict(
            speaker_encoder_state_dict, strict=True
        )
        logger.info("Speaker Encoder is converted.")
    else:
        logger.info("Speaker Encoder not found in the checkpoint, set to zero.")
        model.model.speaker_encoder.embedding.weight.data.zero_()

    torch.save(model.state_dict(), output_path)

    logger.info("All components are converted.")
    logger.info(f"Saved to {output_path}")


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
