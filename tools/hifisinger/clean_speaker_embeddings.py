import torch

# No need to keep optimizer states
data = torch.load(
    "logs/HiFiSVC/cyrj7ncw/checkpoints/epoch=132-step=540000-valid_loss=0.82.ckpt",
    map_location="cpu",
)["state_dict"]

# Remove speaker embeddings
del data["generator.speaker_encoder.embedding.weight"]

torch.save(data, "checkpoints/hifisinger-pretrained-20230329.ckpt")
print("Done")
