import matplotlib
import torch
from matplotlib import pyplot as plt
from fish_audio_preprocess.utils.loudness_norm import loudness_norm

matplotlib.use("Agg")


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def viz_synth_sample(
    gt_mel,
    gt_pitch,
    predict_mel,
    predict_mel_len,
    vocoder,
):
    mel_len = predict_mel_len.item()
    pitch = gt_pitch[:mel_len]
    mel_target = gt_mel[:mel_len].float().detach().T
    mel_prediction = predict_mel[:mel_len].float().detach().T

    fig_mels = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ],
        ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
    )

    wav_reconstruction = vocoder.spec2wav(mel_target, pitch)
    wav_prediction = vocoder.spec2wav(mel_prediction, pitch)

    wav_reconstruction = loudness_norm(wav_reconstruction.cpu().float().numpy(), 44100)
    wav_prediction = loudness_norm(wav_prediction.cpu().float().numpy(), 44100)

    wav_reconstruction = torch.from_numpy(wav_reconstruction)
    wav_prediction = torch.from_numpy(wav_prediction)

    return fig_mels, wav_reconstruction, wav_prediction


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)

    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()

    if isinstance(f0_gt, torch.Tensor):
        f0_gt = f0_gt.detach().cpu().numpy()

    plt.plot(f0_gt, color="r", label="gt")

    if f0_cwt is not None:
        if isinstance(f0_cwt, torch.Tensor):
            f0_cwt = f0_cwt.detach().cpu().numpy()

        plt.plot(f0_cwt, color="b", label="cwt")

    if f0_pred is not None:
        if isinstance(f0_pred, torch.Tensor):
            f0_pred = f0_pred.detach().cpu().numpy()

        plt.plot(f0_pred, color="green", label="pred")

    plt.legend()

    return fig
