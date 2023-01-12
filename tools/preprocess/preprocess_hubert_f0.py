import argparse
import json

# import h5py
import logging
import os
import random
from glob import glob

import torch
from pyworld import pyworld
from scipy.io import wavfile
from tqdm import tqdm
import harmof0
import torchcrepe
from sklearn.cluster import KMeans
from audio.tools import get_mel_from_wav

from diff_svc.utils.stft import TacotronSTFT
from diff_svc.feature_extractors.wav2vec2_xlsr import Wav2Vec2XLSR
import utils.tools
from utils import mel_spectrogram_torch
from utils.tools import get_configs_of

logging.getLogger("numba").setLevel(logging.WARNING)

import librosa
import numpy as np
import parselmouth

dev = "cuda" if torch.cuda.is_available() else "cpu"
config, *_ = get_configs_of("ms")
sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
hop_length = config["preprocessing"]["stft"]["hop_length"]
STFT = TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    config["preprocessing"]["stft"]["hop_length"],
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    config["preprocessing"]["audio"]["sampling_rate"],
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)


def get_f0(path, p_len=None, f0_up_key=0):
    x, sr = librosa.load(path, sr=None)
    assert sr == sampling_rate
    if p_len is None:
        p_len = x.shape[0] // hop_length
    else:
        assert abs(p_len - x.shape[0] // hop_length) < 3, (path, p_len, x.shape)
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = (
        parselmouth.Sound(x, sampling_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")

    f0bak = f0.copy()
    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0bak


def resize2d(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * target_len, len(source)) / target_len,
        np.arange(0, len(source)),
        source,
    )
    res = np.nan_to_num(target)
    return res


def compute_f0(path, c_len, threshold=0.05):
    x, sr = librosa.load(path, sr=None)
    assert sr == sampling_rate

    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * hop_length / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, sampling_rate)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    assert abs(c_len - x.shape[0] // hop_length) < 3, (c_len, f0.shape)

    # print(f0)

    # wav16k = librosa.resample(x, sr, 16000)
    # wav16k_torch = torch.FloatTensor(wav16k).to(dev)[None]

    # # 频率范围
    # f0_min = 40.0
    # f0_max = 1100.0

    # # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    # f0, pd = torchcrepe.predict(
    #     wav16k_torch,
    #     16000,
    #     80,
    #     f0_min,
    #     f0_max,
    #     pad=True,
    #     model="full",
    #     batch_size=1024,
    #     device=dev,
    #     return_periodicity=True,
    # )

    # # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    # pd = torchcrepe.filter.median(pd, 3)
    # pd = torchcrepe.threshold.Silence(-60.0)(pd, wav16k_torch, 16000, 80)
    # f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    # f0 = torchcrepe.filter.mean(f0, 3)

    # # 将nan频率（uv部分）转换为0频率
    # f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # # 去掉0频率，并线性插值
    # nzindex = torch.nonzero(f0[0]).squeeze()
    # f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    # time_org = 0.005 * nzindex.cpu().numpy()
    # time_frame = np.arange(c_len) * 512 / 44100
    # if f0.shape[0] == 0:
    #     f0 = torch.FloatTensor(time_frame.shape[0]).fill_(0)
    # else:
    #     f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

    # print(f0)
    # exit()
    return None, resize2d(f0, c_len)


def process(filename):
    print(filename)

    mel_path = filename + ".mel.npy"
    if not os.path.exists(mel_path):
        wav, sr = librosa.load(filename, sr=None)
        assert sr == sampling_rate
        mel_spectrogram, energy = get_mel_from_wav(wav, STFT)
        np.save(mel_path, mel_spectrogram)
    else:
        mel_spectrogram = np.load(mel_path)

    save_name = filename + ".soft.npy"
    if not os.path.exists(save_name) or True:
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav, sr = librosa.load(filename, sr=16000)
        assert sr == 16000
        wav = torch.from_numpy(wav).to(devive)
        with torch.no_grad():
            c = hmodel(wav, 16000).cpu().squeeze(0)
        c = utils.tools.repeat_expand_2d(c, mel_spectrogram.shape[-1]).numpy()
        np.save(save_name, c)
    else:
        c = np.load(save_name)

    # wav, _ = librosa.load(filename + ".22k.wav", sr=22050)
    # wav = torch.from_numpy(wav).unsqueeze(0).to(dev)
    # sr_mel = mel_spectrogram_torch(wav, 1024, 80, 22050, 256, 1024, 0, 8000)

    # samples = random.choices(range(68, 92 + 1), k=n_sr)
    # for i in range(n_sr):
    #     mel_rs = utils.transform(sr_mel, samples[i])
    #     wav_rs = vocoder(mel_rs)[0][0].detach().cpu().numpy()
    #     _wav_rs = librosa.resample(wav_rs, orig_sr=22050, target_sr=16000)
    #     wav_rs = torch.from_numpy(_wav_rs).to(dev)
    #     # c = utils.tools.get_cn_hubert_units(hmodel, wav_rs).cpu().squeeze(0)
    #     c = hmodel(wav_rs, 16000).cpu().squeeze(0)
    #     c = utils.tools.repeat_expand_2d(c, mel_spectrogram.shape[-1]).numpy()
    #     np.save(save_name.replace(".soft.npy", f".{i}.soft.npy"), c)

    # c = np.load(save_name.replace(".soft.npy", f".0.soft.npy"))

    f0path = filename + ".f0.npy"
    if not os.path.exists(f0path) or True:
        cf0, f0 = compute_f0(filename, c.shape[-1])
        np.save(f0path, f0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset/", help="path to input dir"
    )
    parser.add_argument("--dataset", type=str, default="ms", help="config dataset")

    args = parser.parse_args()
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    n_sr = int(preprocess_config["preprocessing"]["n_sr"])
    print("Loading hubert for content...")
    # hmodel = utils.tools.load_cn_model(0 if torch.cuda.is_available() else None)
    hmodel = Wav2Vec2XLSR()
    hmodel.eval()
    hmodel.to(dev)

    pit = harmof0.PitchTracker(hop_length=80, post_processing=False, low_threshold=0.2)

    print("Loaded hubert.")
    vocoder = utils.tools.get_vocoder(0 if torch.cuda.is_available() else None)

    filenames = glob(f"{args.in_dir}/**/*.wav", recursive=True)  # [:10]
    filenames = [i for i in filenames if not i.endswith((".22k.wav", ".16k.wav"))]

    for filename in tqdm(filenames):
        process(filename)
