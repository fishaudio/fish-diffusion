import json
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .builder import DENOISERS, DIFFUSIONS
from .noise_predictor import (
    NaiveNoisePredictor,
    PLMSNoisePredictor,
    UNIPCNoisePredictor,
)


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


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


@DIFFUSIONS.register_module()
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoiser,
        mel_channels=128,
        noise_schedule="linear",
        timesteps=1000,
        max_beta=0.01,
        s=0.008,
        noise_loss="l1",
        sampler_interval=10,
        spec_stats_path="dataset/stats.json",
        spec_min=None,
        spec_max=None,
        noise_predictor=None,
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

        # calculations for diffusion q(x_t | x_{t-1}) and others
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
        self.unipc_noise_predictor = UNIPCNoisePredictor(betas=betas)

        if noise_predictor is None:
            noise_predictor = "naive" if sampler_interval == 1 else "unipc"

        self.noise_predictor = noise_predictor

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, x_masks=None, cond_masks=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        noised_mel = self.q_sample(x_start=x_start, t=t, noise=noise)
        epsilon = self.denoise_fn(noised_mel, t, cond)

        if x_masks is not None:
            # Apply mask
            noised_mel = noised_mel.masked_fill(x_masks[:, None, :], 0.0)
            epsilon = epsilon.masked_fill(x_masks[:, None, :], 0.0)

        if cond_masks is not None:
            # Apply mask
            cond = cond.masked_fill(cond_masks[:, None, :], 0.0)

        loss = self.get_mel_loss(self.noise_loss, noise, epsilon)

        noised_mel, epsilon = noised_mel.squeeze(1).transpose(1, 2), epsilon.squeeze(
            1
        ).transpose(1, 2)

        return noised_mel, epsilon, loss

    def get_mel_loss(self, loss_fn, noise, epsilon):
        if isinstance(loss_fn, list):
            loss = sum(
                self.get_mel_loss(loss_fn, noise, epsilon) * weight
                for weight, loss_fn in loss_fn
            )
        elif loss_fn == "l1":
            loss = F.l1_loss(noise, epsilon)
        elif loss_fn == "smoothed-l1":
            loss = F.smooth_l1_loss(noise, epsilon)
        elif loss_fn == "l2":
            loss = F.mse_loss(noise, epsilon)
        elif callable(loss_fn):
            loss = loss_fn(noise, epsilon)
        else:
            raise NotImplementedError()

        return loss

    def train_step(self, features, mel, x_masks=None, cond_masks=None):
        # Cond 基本就是 hubert / fs2 参数
        b, *_, device = *features.shape, features.device
        features = features.transpose(1, 2)

        # 计算损失
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = self.norm_spec(mel).transpose(1, 2)  # [B, M, T]

        noised_mels, epsilon, loss = self.p_losses(
            x, t, features, x_masks=x_masks, cond_masks=cond_masks
        )

        return dict(
            loss=loss,
            noised_mels=noised_mels,
            epsilon=epsilon,
            t=t,
        )

    @torch.jit.unused
    def _use_tqdm(self, chunks):
        return tqdm(chunks)

    @torch.jit.script_if_tracing
    def forward(
        self,
        features,
        sampler_interval=None,
        progress: bool = False,
        skip_steps: int = 0,
        original_mel: torch.Tensor = None,
        noise_predictor: str = None,
        x_masks: torch.Tensor = None,
        cond_masks: torch.Tensor = None,
    ):
        if sampler_interval is None:
            sampler_interval = self.sampler_interval

        if noise_predictor is None:
            noise_predictor = self.noise_predictor

        noise_predictor = noise_predictor.lower()

        device = features.device
        features = features.transpose(1, 2)

        if original_mel is None:
            temp = features if x_masks is None else x_masks
            shape = (temp.shape[0], self.mel_bins, temp.shape[-1])
            x = torch.randn(shape, device=device)
        else:
            x = self.norm_spec(original_mel)
            shape = x.shape

        if skip_steps:
            # Apply noise for skip_steps
            t = torch.tensor(
                [self.num_timesteps - skip_steps], device=device, dtype=torch.long
            )
            x = self.q_sample(x_start=x, t=t, noise=torch.randn_like(x))

        chunks = torch.arange(
            0,
            self.num_timesteps - skip_steps,
            sampler_interval,
            dtype=torch.long,
            device=device,
        ).flip(0)[:, None]

        if progress and noise_predictor in ("naive", "plms"):
            chunks = self._use_tqdm(chunks)

        # If using naive sampling
        if noise_predictor == "naive":
            for t in chunks:
                noise = self.denoise_fn(
                    x, t, features, x_masks=x_masks, cond_masks=cond_masks
                )
                x = self.naive_noise_predictor(x=x, t=t, noise=noise)

            return self.denorm_spec(x.transpose(1, 2))

        if noise_predictor == "unipc":
            # UNIPC sampling
            x = self.unipc_noise_predictor(
                denoise_fn=self.denoise_fn,
                x=x,
                cond=features,
                progress=progress,
                sampler_interval=sampler_interval,
                x_masks=x_masks,
                cond_masks=cond_masks,
            )

            return self.denorm_spec(x.transpose(1, 2))

        if noise_predictor == "plms":
            # Hard part: PLMS sampling
            # Credit: OpenVPI's implementation

            plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
            noise_list = torch.zeros((0, *shape), device=device)

            for t in chunks:
                noise_pred = self.denoise_fn(
                    x, t, features, x_masks=x_masks, cond_masks=cond_masks
                )
                t_prev = t - sampler_interval
                t_prev = t_prev * (t_prev > 0)

                if plms_noise_stage == 0:
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

            return self.denorm_spec(x.transpose(1, 2))

        raise NotImplementedError(f"Unknown noise predictor: {noise_predictor}")

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
