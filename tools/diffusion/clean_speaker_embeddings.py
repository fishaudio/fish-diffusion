import torch

data = torch.load(
    "checkpoints/epoch=198-step=260000-valid_loss=0.18.ckpt", map_location="cpu"
)

del data["state_dict"]["model.speaker_encoder.embedding.weight"]

torch.save(data, "checkpoints/epoch=198-step=260000-valid_loss=0.18-fixed.ckpt")
