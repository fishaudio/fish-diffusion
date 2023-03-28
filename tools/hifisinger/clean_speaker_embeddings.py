import torch

# No need to keep optimizer states
data = torch.load(
    "logs/HiFiSVC/version_None/checkpoints/epoch=122-step=500000-valid_loss=0.86.ckpt",
    map_location="cpu",
)["state_dict"]

# Remove speaker embeddings
del data["generator.speaker_encoder.embedding.weight"]

torch.save(data, "checkpoints/hifisinger-demo-20230328.ckpt")
print("Done")
