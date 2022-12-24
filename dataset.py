import json
import math
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
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

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        # pitch stats
        self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
        self.f0_mean = float(preprocess_config["preprocessing"]["pitch"]["f0_mean"])
        self.f0_std = float(preprocess_config["preprocessing"]["pitch"]["f0_std"])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        f0_path = os.path.join(
            self.preprocessed_path,
            "f0",
            "{}-f0-{}.npy".format(speaker, basename),
        )
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0, self.preprocess_config["preprocessing"]["pitch"])
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        mel2ph_path = os.path.join(
            self.preprocessed_path,
            "mel2ph",
            "{}-mel2ph-{}.npy".format(speaker, basename),
        )
        mel2ph = np.load(mel2ph_path)

        cwt_spec = f0_mean = f0_std = f0_ph = None
        if self.pitch_type == "cwt":
            cwt_spec_path = os.path.join(
                self.preprocessed_path,
                "cwt_spec",
                "{}-cwt_spec-{}.npy".format(speaker, basename),
            )
            cwt_spec = np.load(cwt_spec_path)
            f0cwt_mean_std_path = os.path.join(
                self.preprocessed_path,
                "f0cwt_mean_std",
                "{}-f0cwt_mean_std-{}.npy".format(speaker, basename),
            )
            f0cwt_mean_std = np.load(f0cwt_mean_std_path)
            f0_mean, f0_std = float(f0cwt_mean_std[0]), float(f0cwt_mean_std[1])
        elif self.pitch_type == "ph":
            f0_phlevel_sum = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.from_numpy(f0).float())
            f0_phlevel_num = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.ones(f0.shape)).clamp_min(1)
            f0_ph = (f0_phlevel_sum / f0_phlevel_num).numpy()

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "f0": f0,
            "f0_ph": f0_ph,
            "uv": uv,
            "cwt_spec": cwt_spec,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "energy": energy,
            "duration": duration,
            "mel2ph": mel2ph,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        f0s = [data[idx]["f0"] for idx in idxs]
        uvs = [data[idx]["uv"] for idx in idxs]
        cwt_specs = f0_means = f0_stds = f0_phs = None
        if self.pitch_type == "cwt":
            cwt_specs = [data[idx]["cwt_spec"] for idx in idxs]
            f0_means = [data[idx]["f0_mean"] for idx in idxs]
            f0_stds = [data[idx]["f0_std"] for idx in idxs]
            cwt_specs = pad_2D(cwt_specs)
            f0_means = np.array(f0_means)
            f0_stds = np.array(f0_stds)
        elif self.pitch_type == "ph":
            f0s = [data[idx]["f0_ph"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel2phs = [data[idx]["mel2ph"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        f0s = pad_1D(f0s)
        uvs = pad_1D(uvs)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        mel2phs = pad_1D(mel2phs)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
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


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)
