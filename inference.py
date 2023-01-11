import argparse
import os
import librosa
import numpy as np
import parselmouth
import soundfile
import torch
from diff_svc.feature_extractors.wav2vec2_xlsr import Wav2Vec2XLSR
from train import DiffSVC

import utils.tools
from utils.tools import get_configs_of

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

    f0 = (
        parselmouth.Sound(x, sample_rate)
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

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0


def getc(filename, hmodel):
    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = librosa.load(filename, sr=16000)
    # wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
    c = hmodel(wav, 16000).cpu().squeeze(0)
    c = utils.tools.repeat_expand_2d(
        c, int((wav.shape[0] * sample_rate / 16000) // hop_len)
    ).numpy()
    return c


if __name__ == "__main__":
    speaker_id = 2
    conf_name = "ms"
    trans = 0
    src = "raw/sliced/【咩栗】要抱抱-/0003.wav"
    restore_step = 45600

    tgt = src.replace(".wav", f"_{speaker_id}_{trans}_{restore_step}step.wav").replace(
        "raw", "results"
    )

    os.makedirs(os.path.dirname(tgt), exist_ok=True)

    
    preprocess_config, model_config, train_config = get_configs_of(conf_name)
    train_config["path"]["ckpt_path"] = "output/ckpt/cn_hubert_sr"
    configs = (preprocess_config, model_config, train_config)
    # train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}".format(args.model)

    model = DiffSVC.load_from_checkpoint("diff-svc/2qx3vhvp/checkpoints/diff-svc-epoch=571-valid_loss=0.05.ckpt").to(device)
    feature_extractor = Wav2Vec2XLSR().to(device)

    ids = [src]
    c = getc(src, feature_extractor).T
    c_lens = np.array([c.shape[0]])
    contents = np.array([c])
    speakers = np.array([speaker_id])
    _, f0 = get_f0(src, c.shape[0], trans)

    features = model.model.forward_features(
        speakers=torch.from_numpy(speakers).to(device),
        contents=torch.from_numpy(contents).to(device),
        src_lens=torch.from_numpy(c_lens).to(device),
        max_src_len=max(c_lens),
        mel_lens=torch.from_numpy(c_lens).to(device),
        max_mel_len=max(c_lens),
        pitches=torch.from_numpy(f0)[None].to(device),
    )

    result = model.model.diffusion.inference(
        features["features"]
    )

    wav_prediction = model.vocoder.spec2wav(
        result[0].T, f0=torch.from_numpy(f0).to(device)
    ).cpu().numpy()

    soundfile.write(tgt, wav_prediction, 44100)
    print(wav_prediction)
