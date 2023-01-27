import torch
from mmengine import Config

from train import FishDiffusion

KEYS_MAPPING = {
    "betas": "diffusion.betas",
}


def convert():
    config = Config.fromfile("configs/svc_hubert_soft_diff_svc.py")
    model = FishDiffusion(config)

    diff_svc_state_dict = torch.load(
        "/home/lengyue/workspace/ml-toys/diff-svc/checkpoints/aria-2h/model_ckpt_steps_218000.ckpt",
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

    print(fs2_state_dict.keys())

    # Fix Pitch Encoder
    model.model.pitch_encoder.embedding.weight.data = fs2_state_dict[
        "pitch_embed.weight"
    ]

    # TODO: Support multi speaker
    model.model.speaker_encoder.embedding.weight.data.zero_()

    torch.save(model.state_dict(), "aria-2h.pth")


if __name__ == "__main__":
    convert()
