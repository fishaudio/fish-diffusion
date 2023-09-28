import json
import os
from pathlib import Path

import librosa
import torch
from datasets import Audio, Dataset
from multiprocess import set_start_method
from transformers import AutoProcessor, AutoTokenizer, EncodecModel

set_start_method("spawn", force=True)

encodec_name = "facebook/encodec_24khz"
encodec_processor = AutoProcessor.from_pretrained(encodec_name)
encodec_model = EncodecModel.from_pretrained(encodec_name)
encodec_model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "checkpoints/baichuan2-7b-base-extend", use_fast=False, trust_remote_code=True
)


def tokenize(text, audio, sr=None, speaker=None, max_length=2048, ignore_index=-100):
    assert sr is None or sr == encodec_processor.sampling_rate

    if isinstance(audio, (str, Path)):
        audio, sr = librosa.load(audio, sr=sr, mono=True)

    prompt = "[INST] "
    if speaker:
        prompt += f"[SPK] {speaker} [/SPK] "
    prompt += f"{text} [/INST] "

    inputs = encodec_processor(
        raw_audio=audio, sampling_rate=sr, return_tensors="pt"
    ).to(encodec_model.device)
    outputs = encodec_model.encode(
        inputs["input_values"], inputs["padding_mask"], bandwidth=1.5, return_dict=True
    )

    assert outputs.audio_codes.dim() == 4  # [batch, channel, codebook, code]
    assert outputs.audio_codes.shape[0] == outputs.audio_codes.shape[1] == 1

    codes = outputs.audio_codes[0, 0, 0, :]
    codes_str = " ".join([f"<encodec_{int(c)}>" for c in codes.tolist()])
    prompt += codes_str

    encoded = tokenizer.encode(prompt, truncation=False, max_length=102400)
    input_ids = [tokenizer.bos_token_id] + encoded + [tokenizer.eos_token_id]

    input_ids = input_ids[:max_length]
    labels = input_ids.copy()
    attention_mask = [1] * len(input_ids)

    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    labels += [ignore_index] * (max_length - len(labels))
    attention_mask += [0] * (max_length - len(attention_mask))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def wrap_tokenize(x):
    device = torch.device("cuda", 0)

    if encodec_model.device != device:
        encodec_model.to(device)

    return tokenize(
        text=x["text"],
        audio=x["path"],
        sr=encodec_processor.sampling_rate,
        speaker=x["speaker"],
    )


def generator_libritts_r():
    base = Path("dataset/tts/LibriTTS_R")

    for i in base.rglob("*.wav"):
        text_file = i.with_suffix(".normalized.txt")
        if not text_file.exists():
            continue

        text = text_file.read_text().strip()

        yield {
            "text": text,
            "speaker": f"libritts_{i.parent.parent.name}",
            "path": str(i),
        }


if __name__ == "__main__":
    # with open("dataset/tts/WenetSpeech/WenetSpeech.json") as f:
    #     dataset = json.load(f)

    dataset = Dataset.from_generator(generator_libritts_r)
    dataset = dataset.map(wrap_tokenize, num_proc=8)
    dataset.save_to_disk("dataset/tts/LibriTTS_R_tokenized")
