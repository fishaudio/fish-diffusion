import torch

input_file = (
    "logs/NSF-HiFiGAN/lpm0f6b1/checkpoints/epoch=54-step=69960-valid_loss=0.22.ckpt"
)
output_file = "nsf_hifigan.pt"

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
