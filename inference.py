import argparse
import os
import librosa
import numpy as np
import parselmouth
import soundfile
import torch
from fish_diffusion.feature_extractors.wav2vec2_xlsr import Wav2Vec2XLSR
from fish_diffusion.feature_extractors.chinese_hubert import ChineseHubert
from tools.preprocess.preprocess_hubert_f0 import compute_f0
from train import DiffSVC

import utils.tools
from utils.tools import get_configs_of

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sample_rate = 44100
hop_len = 512


def getc(filename, hmodel):
    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = librosa.load(filename, sr=16000)
    wav = torch.from_numpy(wav).to(devive)
    c = hmodel(wav, 16000).cpu().squeeze(0)
    c = utils.tools.repeat_expand_2d(
        c, int((wav.shape[0] * sample_rate / 16000) // hop_len)
    ).numpy()

    return c


if __name__ == "__main__":
    speaker_id = 0
    conf_name = "ms"
    trans = 0
    src = "raw/sliced/大喜 - 泠鸢yousa,音阙诗听/0001.wav"
    # src = "dataset/aria/斯卡布罗干音_0000/0000.wav"
    restore_step = 45600

    tgt = (
        src.replace(".wav", f"_{speaker_id}_{trans}_{restore_step}step.wav")
        .replace("raw", "results")
        .replace("dataset", "results")
    )

    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    # train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}".format(args.model)

    model = DiffSVC.load_from_checkpoint(
        "logs/diff-svc/1yhuw4kh/checkpoints/diff-svc-epoch=9117-valid_loss=0.11.ckpt",
    ).to(device)

    model.eval()
    # feature_extractor = Wav2Vec2XLSR().to(device)
    feature_extractor = Wav2Vec2XLSR().to(device)
    feature_extractor.eval()

    ids = [src]
    c = getc(src, feature_extractor).T
    c_lens = np.array([c.shape[0]])
    contents = np.array([c])
    speakers = np.array([speaker_id])
    _, f0 = compute_f0(src, c.shape[0])

    f0 *= pow(2, trans / 12)

    print(f0.shape, c.shape)

    # f0 = np.load("dataset/aria/斯卡布罗干音_0000/0000.wav.f0.npy")
    # c = np.load("dataset/aria/斯卡布罗干音_0000/0000.wav.0.soft.npy").T
    # c_lens = np.array([c.shape[0]])
    # contents = np.array([c])

    print(f0.shape, c.shape)

    with torch.no_grad():
        features = model.model.forward_features(
            speakers=torch.from_numpy(speakers).to(device),
            contents=torch.from_numpy(contents).to(device),
            src_lens=torch.from_numpy(c_lens).to(device),
            max_src_len=max(c_lens),
            mel_lens=torch.from_numpy(c_lens).to(device),
            max_mel_len=max(c_lens),
            pitches=torch.from_numpy(f0)[None].to(device),
        )

        result = model.model.diffusion.inference(features["features"])

    from matplotlib import pyplot as plt

    plt.imshow(result[0].T.cpu().numpy())
    plt.savefig("result.png")

    wav_prediction = (
        model.vocoder.spec2wav(result[0].T, f0=torch.from_numpy(f0).to(device))
        .cpu()
        .numpy()
    )

    soundfile.write(tgt, wav_prediction, 44100)
    print(wav_prediction)
