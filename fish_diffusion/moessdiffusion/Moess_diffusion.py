from fish_diffusion.denoisers import DENOISERS
from fish_diffusion.diffusions.noise_predictor import (
    NaiveNoisePredictor,
    PLMSNoisePredictor,
)
import numpy as np
import torch
from torch import nn
import json
from functools import partial
from fish_diffusion.moessdiffusion import MOESSDIFFUSIONS


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


def predict_stage0(noise_pred, noise_pred_prev):
    return (noise_pred + noise_pred_prev) / 2


def predict_stage1(noise_pred, noise_list):
    return (noise_pred * 3
            - noise_list[-1]) / 2


def predict_stage2(noise_pred, noise_list):
    return (noise_pred * 23
            - noise_list[-1] * 16
            + noise_list[-2] * 5) / 12


def predict_stage3(noise_pred, noise_list):
    return (noise_pred * 55
            - noise_list[-1] * 59
            + noise_list[-2] * 37
            - noise_list[-3] * 9) / 24


class AfterDiffusion(nn.Module):
    def __init__(self, spec_max, spec_min):
        super().__init__()
        self.spec_max = spec_max
        self.spec_min = spec_min

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        mel_out = x * d + m
        mel_out = mel_out * 2.30259
        return mel_out.transpose(2, 1)


@MOESSDIFFUSIONS.register_module()
class GaussianDiffusion(nn.Module):
    def __init__(
            self, denoiser, mel_channels=128, noise_schedule="linear", timesteps=1000, max_beta=0.01, s=0.008,
            noise_loss="l1", sampler_interval=10, spec_stats_path="dataset/stats.json", spec_min=None, spec_max=None,
    ):
        super().__init__()
        self.denoise_fn = DENOISERS.build(denoiser)
        self.mel_bins = mel_channels
        betas = get_noise_schedule_list(noise_schedule, timesteps, max_beta, s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.noise_loss = noise_loss
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        assert (spec_min is None and spec_max is None) or (
                spec_min is not None and spec_max is not None
        ), "spec_min and spec_max must be both None or both not None"
        if spec_min is None:
            with open(spec_stats_path) as f:
                stats = json.load(f)

            spec_min = stats["spec_min"]
            spec_max = stats["spec_max"]
        assert (
                len(spec_min) == len(spec_max) == mel_channels
                or len(spec_min) == len(spec_max) == 1
        ), "spec_min and spec_max must be either of length 1 or mel_channels"
        self.register_buffer("spec_min", torch.FloatTensor(spec_min).view(1, 1, -1))
        self.register_buffer("spec_max", torch.FloatTensor(spec_max).view(1, 1, -1))
        self.sampler_interval = sampler_interval
        self.naive_noise_predictor = NaiveNoisePredictor(betas=betas)
        self.plms_noise_predictor = PLMSNoisePredictor(betas=betas)
        self.ad = AfterDiffusion(spec_max=self.spec_max, spec_min=self.spec_min)

    def MoeSSOnnxExport(self, project_name, device):
        sampler_interval = self.sampler_interval
        features = torch.zeros([1, 256, 10]).to(device)
        shape = (features.shape[0], 1, self.mel_bins, features.shape[2])
        x = torch.randn(shape, device=device)
        pndms = 100
        n_frames = features.shape[2]
        step_range = torch.arange(0, 1000, pndms, dtype=torch.long, device=device).flip(0)[:, None]
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
        ot = step_range[0]
        torch.onnx.export(
            self.denoise_fn,
            (x.to(device), ot.to(device), features.to(device)),
            f"{project_name}_denoise.onnx",
            input_names=["noise", "time", "condition"],
            output_names=["noise_pred"],
            dynamic_axes={
                "noise": [3],
                "condition": [2]
            },
            opset_version=16
        )
        for t in step_range:
            noise_pred = self.denoise_fn(x, t, features)
            t_prev = t - sampler_interval
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                torch.onnx.export(
                    self.plms_noise_predictor,
                    (x.to(device), noise_pred.to(device), t.to(device), t_prev.to(device)),
                    f"{project_name}_pred.onnx",
                    input_names=["noise", "noise_pred", "time", "time_prev"],
                    output_names=["noise_pred_o"],
                    dynamic_axes={
                        "noise": [3],
                        "noise_pred": [3]
                    },
                    opset_version=16
                )
                x_pred = self.plms_noise_predictor(x, noise_pred, t, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, features)
                noise_pred_prime = self.plms_noise_predictor.predict_stage0(
                    noise_pred, noise_pred_prev
                )
            elif plms_noise_stage == 1:
                noise_pred_prime = self.plms_noise_predictor.predict_stage1(
                    noise_pred, noise_list
                )
            elif plms_noise_stage == 2:
                noise_pred_prime = self.plms_noise_predictor.predict_stage2(
                    noise_pred, noise_list
                )
            else:
                noise_pred_prime = self.plms_noise_predictor.predict_stage3(
                    noise_pred, noise_list
                )

            noise_pred = noise_pred.unsqueeze(0)
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1
            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)

            x = self.plms_noise_predictor(x, noise_pred_prime, t, t_prev)
        torch.onnx.export(
            self.ad,
            x.to(device),
            f"{project_name}_after.onnx",
            input_names=["x"],
            output_names=["mel_out"],
            dynamic_axes={
                "x": [3]
            },
            opset_version=16
        )
