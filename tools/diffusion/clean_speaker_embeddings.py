import torch

data = torch.load(
    "logs/DiffSVC/9ddsi2gk/checkpoints/epoch=88-step=300000-valid_loss=0.08.ckpt",
    map_location="cpu",
)

del data["state_dict"]["model.speaker_encoder.embedding.weight"]

torch.save(data, "checkpoints/content-vec-pretrained-v1.ckpt")
