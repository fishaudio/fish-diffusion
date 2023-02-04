# This code is for developers only.

import torch
from train_whisper import WhisperModel

model = WhisperModel.load_from_checkpoint(
    "logs/whisper/d2mevxji/checkpoints/epoch=49-val_loss=0.0792-val_acc=0.8725.ckpt"
)
model.model.eval()
model.model.save("checkpoints/aligned-whisper-cn-25k-v1.ckpt")

state_dict = torch.load("checkpoints/aligned-whisper-cn-25k-v1.ckpt")

print(state_dict["model_state_dict"].keys())
