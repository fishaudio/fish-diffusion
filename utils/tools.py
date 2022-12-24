import os
import json
import yaml
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.pitch_tools import denorm_f0, expand_f0_ph, cwt2f0


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def to_device(data, device):
    if len(data) == 18:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device) if mels is not None else mels
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).long().to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        cwt_specs = torch.from_numpy(cwt_specs).float().to(device) if cwt_specs is not None else cwt_specs
        f0_means = torch.from_numpy(f0_means).float().to(device) if f0_means is not None else f0_means
        f0_stds = torch.from_numpy(f0_stds).float().to(device) if f0_stds is not None else f0_stds
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        mel2phs = torch.from_numpy(mel2phs).long().to(device)

        pitch_data = {
            "pitch": pitches,
            "f0": f0s,
            "uv": uvs,
            "cwt_spec": cwt_specs,
            "f0_mean": f0_means,
            "f0_std": f0_stds,
        }

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitch_data,
            energies,
            durations,
            mel2phs,
        ]

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, lr=None, figs=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/noise_loss", losses[2], step)
        for k, v in losses[3].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        for k, v in losses[5].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if figs is not None:
        for k, v in figs.items():
            logger.add_figure("{}/{}".format(tag, k), v, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(args, targets, predictions, vocoder, model_config, preprocess_config, diffusion):

    pitch_config = preprocess_config["preprocessing"]["pitch"]
    pitch_type = pitch_config["pitch_type"]
    use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
    use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
    basename = targets[0][0]
    src_len = predictions[10][0].item()
    mel_len = predictions[11][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    duration = targets[11][0, :src_len].int().detach().cpu().numpy()
    figs = {}
    if use_pitch_embed:
        pitch_prediction, pitch_target = predictions[4], targets[9]
        f0 = pitch_target["f0"]
        if pitch_type == "ph":
            mel2ph = targets[12]
            f0 = expand_f0_ph(f0, mel2ph, pitch_config)
            f0_pred = expand_f0_ph(pitch_prediction["pitch_pred"][:, :, 0], mel2ph, pitch_config)
            figs["f0"] = f0_to_figure(f0[0, :mel_len], None, f0_pred[0, :mel_len])
        else:
            f0 = denorm_f0(f0, pitch_target["uv"], pitch_config)
            if pitch_type == "cwt":
                # cwt
                cwt_out = pitch_prediction["cwt"]
                cwt_spec = cwt_out[:, :, :10]
                cwt = torch.cat([cwt_spec, pitch_target["cwt_spec"]], -1)
                figs["cwt"] = spec_to_figure(cwt[0, :mel_len])
                # f0
                f0_pred = cwt2f0(cwt_spec, pitch_prediction["f0_mean"], pitch_prediction["f0_std"], pitch_config["cwt_scales"])
                if pitch_config["use_uv"]:
                    assert cwt_out.shape[-1] == 11
                    uv_pred = cwt_out[:, :, -1] > 0
                    f0_pred[uv_pred > 0] = 0
                f0_cwt = denorm_f0(pitch_target["f0_cwt"], pitch_target["uv"], pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], f0_cwt[0, :mel_len], f0_pred[0, :mel_len])
            elif pitch_type == "frame":
                # f0
                uv_pred = pitch_prediction["pitch_pred"][:, :, 1] > 0
                pitch_pred = denorm_f0(pitch_prediction["pitch_pred"][:, :, 0], uv_pred, pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], None, pitch_pred[0, :mel_len])
    if use_energy_embed:
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy_prediction = predictions[5][0, :src_len].detach().cpu().numpy()
            energy_prediction = expand(energy_prediction, duration)
            energy_target = targets[10][0, :src_len].detach().cpu().numpy()
            energy_target = expand(energy_target, duration)
        else:
            energy_prediction = predictions[5][0, :mel_len].detach().cpu().numpy()
            energy_target = targets[10][0, :mel_len].detach().cpu().numpy()
        figs["energy"] = energy_to_figure(energy_target, energy_prediction)

    if args.model == "aux":
        mel_prediction = predictions[0][0, :mel_len].float().detach().transpose(0, 1)
    else:
        # diffusion_step = predictions[3][0].item()
        # noisy_mels = predictions[0][0, :mel_len].detach().transpose(0, 1)
        # noise_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
        mel_prediction = diffusion.sampling()[0, :mel_len].detach().transpose(0, 1)
        diffusion.aux_mel = None

    figs["mel"] = plot_mel(
        [
            # noisy_mels.cpu().numpy(),
            # noise_prediction.cpu().numpy(),
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ],
        # [f"Diffused Spectrogram at {diffusion_step}-step", f"Epsilon Prediction from {diffusion_step}-step", "Sampled Spectrogram", "Ground-Truth Spectrogram"],
        ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return figs, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    teacher_forced_tag = "_teacher_forced" if args.teacher_forced else ""
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[10][i].item()
        mel_len = predictions[11][i].item()
        mel_prediction = predictions[0][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[7][i, :src_len].detach().cpu().numpy()

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}{}.png".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.png".format(basename, teacher_forced_tag))
        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(fig_save_dir)
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[0].transpose(1, 2)
    lengths = predictions[11] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}{}.wav".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.wav".format(basename, teacher_forced_tag)),
            sampling_rate, wav)


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


def energy_to_figure(energy_gt, energy_pred=None):
    fig = plt.figure()
    if isinstance(energy_gt, torch.Tensor):
        energy_gt = energy_gt.detach().cpu().numpy()
    plt.plot(energy_gt, color="r", label="gt")
    if energy_pred is not None:
        if isinstance(energy_pred, torch.Tensor):
            energy_pred = energy_pred.detach().cpu().numpy()
        plt.plot(energy_pred, color="green", label="pred")
    plt.legend()
    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_noise_schedule_list(schedule_mode, timesteps, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError
    return schedule_list


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)
