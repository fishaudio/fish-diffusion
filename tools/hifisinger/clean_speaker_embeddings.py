import torch

data = torch.load(
    "logs/HiFiSVC/5cdrm2ww/checkpoints/epoch=138-step=300000-valid_loss=0.85.ckpt",
    map_location="cpu",
)

del data["state_dict"]["generator.speaker_encoder.embedding.weight"]

torch.save(data, "checkpoints/hifisinger-lengyue-pretrained-v1.ckpt")
print("Done")
