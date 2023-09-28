import json
from pathlib import Path

import librosa
import soundfile as sf
import torch
import torchaudio
from fish_audio_preprocess.utils.separate_audio import init_model
from tqdm import tqdm

from fish_diffusion.utils.audio import separate_vocals

meta_path = Path("dataset/tts/WenetSpeech/WenetSpeech.json")
dataset_path = Path("dataset/tts/WenetSpeech")
cleaned_path = Path("dataset/tts/WenetSpeech/cleaned")
if not cleaned_path.exists():
    cleaned_path.mkdir(parents=True)

demucs = init_model("htdemucs", "cuda:1")
print("Model loaded")

with open(meta_path) as f:
    dataset = json.load(f)["audios"]

print("Dataset loaded")

for audio in tqdm(dataset):
    raw_audio, sr = librosa.load(dataset_path / audio["path"], sr=24000)

    for idx, segment in enumerate(audio["segments"]):
        if segment["confidence"] <= 0.95:
            continue

        # Load audio
        begin = int(segment["begin_time"] * sr)
        end = int(segment["end_time"] * sr)
        segment_audio = raw_audio[begin:end]

        # Demucs separate
        segment_audio, _ = separate_vocals(segment_audio, sr, separate_model=demucs)

        # Write audio
        temp_path = cleaned_path / audio["aid"] / f"S{idx:05d}.flac"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temp_path, segment_audio, samplerate=sr)

        # Write text
        temp_path = temp_path.with_suffix(".txt")
        temp_path.write_text(segment["text"])

print("Done")
