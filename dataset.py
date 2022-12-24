import json
import math
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.tools import pad_1D, pad_2D
from utils.pitch_tools import norm_interp_f0, get_lf0_cwt


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.preprocess_config = preprocess_config
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.wavpaths = self.process_meta(
            filename
        )
        with open(os.path.join("dataset", "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        # pitch stats
        self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
        self.f0_mean = float(preprocess_config["preprocessing"]["pitch"]["f0_mean"])
        self.f0_std = float(preprocess_config["preprocessing"]["pitch"]["f0_std"])

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):
        wavpath = self.wavpaths[idx]
        speaker = wavpath.split(os.sep)[-2]
        speaker_id = self.speaker_map[speaker]

        mel_path = wavpath + ".mel.npy"
        mel = np.load(mel_path).T

        c_path = wavpath + ".soft.npy"
        c = np.load(c_path).T
        pitch_path = wavpath + ".f0.npy"
        pitch = np.load(pitch_path)

        sample = {
            "id": wavpath,
            "speaker": speaker_id,
            "content": c,
            "mel": mel,
            "pitch": pitch,
        }

        return sample

    def process_meta(self, filename):
        with open(
            filename, "r", encoding="utf-8"
        ) as f:
            wavpaths = []
            for line in f.readlines():
                wavpath = line.strip("\n")
                wavpaths.append(wavpath)
            return wavpaths

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        contents = [data[idx]["content"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])

        c_lens = np.array([c.shape[0] for c in contents])

        speakers = np.array(speakers)
        mels = pad_2D(mels)
        contents = pad_2D(contents)
        pitches = pad_1D(pitches)

        return (
            ids,
            speakers,
            contents,
            c_lens,
            max(c_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
        )

    def collate_fn(self, data):
        data_size = len(data)
        idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

#
# class TextDataset(Dataset):
#     def __init__(self, filepath, preprocess_config):
#         self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
#
#         self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
#             filepath
#         )
#         with open(
#             os.path.join(
#                 preprocess_config["path"]["preprocessed_path"], "speakers.json"
#             )
#         ) as f:
#             self.speaker_map = json.load(f)
#
#     def __len__(self):
#         return len(self.text)
#
#     def __getitem__(self, idx):
#         basename = self.basename[idx]
#         speaker = self.speaker[idx]
#         speaker_id = self.speaker_map[speaker]
#         raw_text = self.raw_text[idx]
#         phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
#
#         return (basename, speaker_id, phone, raw_text)
#
#     def process_meta(self, filename):
#         with open(filename, "r", encoding="utf-8") as f:
#             name = []
#             speaker = []
#             text = []
#             raw_text = []
#             for line in f.readlines():
#                 n, s, t, r = line.strip("\n").split("|")
#                 name.append(n)
#                 speaker.append(s)
#                 text.append(t)
#                 raw_text.append(r)
#             return name, speaker, text, raw_text
#
#     def collate_fn(self, data):
#         ids = [d[0] for d in data]
#         speakers = np.array([d[1] for d in data])
#         texts = [d[2] for d in data]
#         raw_texts = [d[3] for d in data]
#         text_lens = np.array([text.shape[0] for text in texts])
#
#         texts = pad_1D(texts)
#
#         return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
