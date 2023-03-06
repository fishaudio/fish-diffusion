import io
import logging
import socket
import time

import librosa
import numpy as np
import torch
from mmengine import Config

from tools.diffusion.inference import SVCInference

# fish下只需传入下列参数，文件路径以项目根目录为准
checkpoint_path = (
    "logs/DiffSVC/9ddsi2gk/checkpoints/epoch=88-step=300000-valid_loss=0.08.ckpt"
)
config_path = "configs/exp_super_mix_v2.py"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config.fromfile(config_path)
model = SVCInference(config, checkpoint_path)
model = model.to(device)


# Create a TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("10.0.0.2", 8081))

# Listen for incoming connections
server.listen()
print("The server is ready to receive")

# Accept incoming connections
connection, client_address = server.accept()
buff = b""
frame_size = 3 * 4 * 44100

# Receive and print the data
while True:
    data = connection.recv(frame_size)
    buff += data

    if len(buff) < frame_size:
        continue

    start_time = time.time()

    data, buff = buff[:frame_size], buff[frame_size:]
    audio = np.frombuffer(data, dtype=np.float32)

    intervals = librosa.effects.split(audio, top_db=10)
    new_audio = np.zeros_like(audio)

    if len(intervals) == 1 and intervals[0][0] == 0 and intervals[0][1] == len(audio):
        connection.sendall(new_audio.tobytes())
        continue

    for start, end in intervals:
        new_audio[start:end] = audio[start:end]

    audio = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Inference
    audio = model.forward(
        audio, 44100, pitch_adjust=4, speaker_id=0, sampler_interval=10
    )

    if len(audio) < frame_size // 4:
        audio = np.pad(audio, (0, frame_size // 4 - len(audio)), "constant")

    data = audio.tobytes()[:frame_size]
    print(f"Time: {time.time() - start_time}")

    # Echo back to client
    connection.sendall(data)
