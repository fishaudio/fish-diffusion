import os
import re
import argparse
from string import punctuation

import parselmouth
from tqdm import tqdm
import librosa
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style
import utils.tools
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize(model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    def synthesize_(batch):
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )[0]
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )

    if args.teacher_forced:
        for batchs_ in batchs:
            for batch in tqdm(batchs_):
                batch = list(batch)
                batch[6] = None # set mel None for diffusion sampling
                synthesize_(batch)
    else:
        for batch in tqdm(batchs):
            synthesize_(batch)


sample_rate = 44100
hop_len = 512

def get_f0(path,p_len=None, f0_up_key=0):
    x, sr = librosa.load(path, sr=None)
    assert sr == sample_rate
    if p_len is None:
        p_len = x.shape[0]//hop_len
    else:
        assert abs(p_len-x.shape[0]//hop_len) < 3, (path, p_len, x.shape)
    time_step = hop_len / sample_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sample_rate).to_pitch_ac(
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


def getc(filename,hmodel):

    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = librosa.load(filename, sr=16000)
    wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
    c = utils.tools.get_hubert_content(hmodel, wav).cpu().squeeze(0)
    c = utils.tools.repeat_expand_2d(c, (wav.shape[0] * sample_rate / 16000) // hop_len).numpy()
    return c

if __name__ == "__main__":
    speaker_id = 0
    conf_name = "ms"
    trans = 0
    src = "君の知らない物語-src.wav"

    tgt = src.replace(".wav", f"_{speaker_id}_{trans}.wav")
    restore_step = 10000
    preprocess_config, model_config, train_config = get_configs_of(conf_name)
    configs = (preprocess_config, model_config, train_config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True,default=restore_step)
    args = parser.parse_args()

    model = get_model(args, configs, device, train=False)
    hmodel = utils.tools.get_hubert_model(0 if torch.cuda.is_available() else None)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    ids = [src]
    c = getc(src, hmodel)
    c_lens = np.array([c.shape[0]])
    
    contents = np.array([c])
    speakers = np.array([speaker_id])
    text_lens = np.array([len(texts[0])])
    
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args, configs, vocoder, batchs, control_values)
