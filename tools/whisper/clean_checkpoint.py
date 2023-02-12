# This code is for developers only.

import torch

from tools.whisper.train import WhisperModel

model = WhisperModel.load_from_checkpoint("logs/whisper/esuhztrl/checkpoints/last.ckpt")
model.model.eval()
model.model.save("checkpoints/aligned-whisper-cn-25k-v1.1.ckpt")

state_dict = torch.load("checkpoints/aligned-whisper-cn-25k-v1.1.ckpt")

print(state_dict["model_state_dict"].keys())
