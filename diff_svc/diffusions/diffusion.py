import json
import os
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from utils.tools import get_noise_schedule_list


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denosier,
        mel_channels=128,
        keep_bins=128,
        noise_schedule="linear",
        timesteps=1000,
        max_beta=0.01,
        s=0.008,
        noise_loss="smoothed-l1",
        spec_stats_path="dataset/stats.json",
    ):
        super().__init__()

        self.denoise_fn = denosier
        self.mel_bins = mel_channels

        betas = get_noise_schedule_list(noise_schedule, timesteps, max_beta, s)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.noise_loss = noise_loss

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        with open(spec_stats_path) as f:
            stats = json.load(f)
            self.register_buffer(
                "spec_min",
                torch.FloatTensor(stats["spec_min"])[None, None, :keep_bins],
            )
            self.register_buffer(
                "spec_max",
                torch.FloatTensor(stats["spec_max"])[None, None, :keep_bins],
            )

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond)
        epsilon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            epsilon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=epsilon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, cond=cond, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(
        self, x, t, interval, cond, clip_denoised=True, repeat_noise=False
    ):
        """
        Use the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(
                self.alphas_cumprod,
                torch.max(t - interval, torch.zeros_like(t)),
                x.shape,
            )
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t
            )
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            t = torch.nn.functional.relu(t - interval)
            noise_pred_prev = self.denoise_fn(x_pred, t, cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (
                23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]
            ) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (
                55 * noise_pred
                - 59 * noise_list[-1]
                + 37 * noise_list[-2]
                - 9 * noise_list[-3]
            ) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        noised_mel = self.q_sample(x_start=x_start, t=t, noise=noise)
        epsilon = self.denoise_fn(noised_mel, t, cond)

        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask[:, None, None, :]

            # Apply mask
            noise = noise.masked_fill(mask, 0.0)
            epsilon = epsilon.masked_fill(mask, 0.0)

        if self.noise_loss == "l1":
            loss = F.l1_loss(noise, epsilon)
        elif self.noise_loss == "smoothed-l1":
            loss = F.smooth_l1_loss(noise, epsilon)
        elif self.noise_loss == "l2":
            loss = F.mse_loss(noise, epsilon)
        elif callable(self.noise_loss):
            loss = self.noise_loss(noise, epsilon)
        else:
            raise NotImplementedError()

        noised_mel, epsilon = noised_mel.squeeze(1).transpose(1, 2), epsilon.squeeze(
            1
        ).transpose(1, 2)

        return noised_mel, epsilon, loss

    def forward(self, features, mel, mel_mask=None):
        # Cond 基本就是 hubert / fs2 参数
        b, *_, device = *features.shape, features.device
        features = features.transpose(1, 2)

        # 计算损失
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = self.norm_spec(mel).transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]

        noised_mels, epsilon, loss = self.p_losses(x, t, features, mask=mel_mask)

        return dict(
            loss=loss,
            noised_mels=noised_mels,
            epsilon=epsilon,
            t=t,
        )

    def inference(self, features):
        # Cond 基本就是 hubert / fs2 参数
        b, *_, device = *features.shape, features.device
        features = features.transpose(1, 2)

        t = self.num_timesteps
        shape = (features.shape[0], 1, self.mel_bins, features.shape[2])
        x = torch.randn(shape, device=device)

        self.noise_list = deque(maxlen=4)
        iteration_interval = 20  # TODO: 读取配置的 PNDM

        for i in tqdm(
            reversed(range(0, t, iteration_interval)),
            desc="sample time step",
            total=t // iteration_interval,
        ):
            x = self.p_sample_plms(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                iteration_interval,
                features,
            )

        x = x[:, 0].transpose(1, 2)

        return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def out2mel(self, x):
        return x
