import argparse

import parselmouth
import soundfile
import librosa
import torch
import numpy as np
import utils.tools
from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sample_rate = 44100
hop_len = 512


def get_f0(path, p_len=None, f0_up_key=0):
    x, sr = librosa.load(path, sr=sample_rate)
    assert sr == sample_rate
    if p_len is None:
        p_len = x.shape[0] // hop_len
    else:
        assert abs(p_len - x.shape[0] // hop_len) < 3, (path, p_len, x.shape)
    time_step = hop_len / sample_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0


def getc(filename, hmodel):
    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = librosa.load(filename, sr=16000)
    wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
    c = utils.tools.get_cn_hubert_units(hmodel, wav).cpu().squeeze(0)
    c = utils.tools.repeat_expand_2d(c, int((wav.shape[1] * sample_rate / 16000) // hop_len)).numpy()
    return c


if __name__ == "__main__":
    speaker_id = 2
    conf_name = "ms"
    trans = -8
    src = "raw/再见.wav"
    restore_step = 45600

    tgt = src.replace(".wav", f"_{speaker_id}_{trans}_{restore_step}step.wav").replace("raw", "results")
    preprocess_config, model_config, train_config = get_configs_of(conf_name)
    train_config["path"]["ckpt_path"] = "output/ckpt/cn_hubert_sr"
    configs = (preprocess_config, model_config, train_config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=False, default=restore_step)
    parser.add_argument("--model", type=str, required=False, default="naive")
    args = parser.parse_args()
    # train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}".format(args.model)

    model = get_model(args, configs, device, train=False)
    hmodel = utils.tools.load_cn_model(0 if torch.cuda.is_available() else None)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    ids = [src]
    c = getc(src, hmodel).T
    c_lens = np.array([c.shape[0]])
    contents = np.array([c])
    speakers = np.array([speaker_id])
    _, f0 = get_f0(src,c.shape[0], trans)

    batch = [ids,speakers, contents, c_lens, max(c_lens), None, c_lens, max(c_lens), np.array([f0])]
    batch = to_device(batch, device)
    output = model(*(batch[1:]))

    mel_len = output[7][0].item()
    pitch = batch[8][0][:mel_len]
    figs = {}
    mel_prediction = output[0][0, :mel_len].detach().transpose(0, 1)

    wav_prediction = vocoder.spec2wav(mel_prediction.cpu().numpy().T, f0=pitch.cpu().numpy())

    soundfile.write(tgt, wav_prediction,44100)
    print(wav_prediction)