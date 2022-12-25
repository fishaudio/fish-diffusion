import os
import argparse

import torch
import json
from glob import glob
import audio as Audio

from pyworld import pyworld
from tqdm import tqdm
from scipy.io import wavfile
from utils.tools import get_configs_of

import utils.tools
#import h5py
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

import parselmouth
import librosa
import numpy as np

config, *_ = get_configs_of("ms")
sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
hop_length = config["preprocessing"]["stft"]["hop_length"]
STFT = Audio.stft.TacotronSTFT(
    config["preprocessing"]["stft"]["filter_length"],
    config["preprocessing"]["stft"]["hop_length"],
    config["preprocessing"]["stft"]["win_length"],
    config["preprocessing"]["mel"]["n_mel_channels"],
    config["preprocessing"]["audio"]["sampling_rate"],
    config["preprocessing"]["mel"]["mel_fmin"],
    config["preprocessing"]["mel"]["mel_fmax"],
)

def get_f0(path,p_len=None, f0_up_key=0):
    x, sr = librosa.load(path, sr=None)
    assert sr == sampling_rate
    if p_len is None:
        p_len = x.shape[0]//hop_length
    else:
        assert abs(p_len-x.shape[0]//hop_length) < 3, (path, p_len, x.shape)
    time_step = hop_length / sampling_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
        f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')

    f0bak = f0.copy()
    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0bak

def resize2d(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res

def compute_f0(path, c_len):
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
    assert abs(c_len - x.shape[0]//hop_length) < 3, (c_len, f0.shape)

    return None, resize2d(f0, c_len)


def process(filename):
    print(filename)

    mel_path = filename + ".mel.npy"
    if not os.path.exists(mel_path):
        wav, sr = librosa.load(filename,sr=None)
        assert sr == sampling_rate
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
        np.save(mel_path, mel_spectrogram)
    else:
        mel_spectrogram = np.load(mel_path)

    save_name = filename+".soft.npy"
    if not os.path.exists(save_name):
        devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav, sr = librosa.load(filename+".16k.wav",sr=None)
        assert sr == 16000
        wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
        c = utils.tools.get_hubert_content(hmodel, wav).cpu().squeeze(0)
        c = utils.tools.repeat_expand_2d(c, mel_spectrogram.shape[-1]).numpy()
        np.save(save_name,c)
    else:
        c = np.load(save_name)

    f0path = filename+".f0.npy"
    if not os.path.exists(f0path):
        cf0, f0 = compute_f0(filename, c.shape[-1])
        np.save(f0path, f0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="dataset/", help="path to input dir")
    args = parser.parse_args()

    print("Loading hubert for content...")
    hmodel = utils.tools.get_hubert_model(0 if torch.cuda.is_available() else None)
    print("Loaded hubert.")

    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)#[:10]
    filenames = [i for i in filenames if not i.endswith(".16k.wav")]
    
    for filename in tqdm(filenames):
        process(filename)
    