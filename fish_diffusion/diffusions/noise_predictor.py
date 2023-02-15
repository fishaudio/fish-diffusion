from functools import partial

import numpy as np
import torch
from torch import nn


def extract(a, t):
    return a[t].reshape((1, 1, 1))


to_torch = partial(torch.tensor, dtype=torch.float32)


class NaiveNoisePredictor(nn.Module):
    def __init__(
        self,
        betas,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        super().__init__()

        # Calculate alphas from loss schedule
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Clip
        self.register_buffer("clip_min", to_torch(clip_min))
        self.register_buffer("clip_max", to_torch(clip_max))

        # Prior and posterior
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t) * x_start
            + extract(self.posterior_mean_coef2, t) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise):
        epsilon = self.predict_start_from_noise(x_t=x, t=t, noise=noise)
        epsilon = torch.clamp(epsilon, min=self.clip_min, max=self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=epsilon, x_t=x, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    def forward(self, x, t, noise):
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, noise)

        noise = torch.randn_like(x)
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1)

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


class PLMSNoisePredictor(nn.Module):
    def __init__(self, betas):
        super().__init__()

        # Calculate alphas from loss schedule
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Prior and posterior
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

    def forward(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * (
            (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
            - 1
            / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
            * noise_t
        )
        x_pred = x + x_delta

        return x_pred

    def predict_stage0(self, noise_pred, noise_pred_prev):
        return (noise_pred + noise_pred_prev) / 2

    def predict_stage1(self, noise_pred, noise_list):
        return (noise_pred * 3 - noise_list[-1]) / 2

    def predict_stage2(self, noise_pred, noise_list):
        return (noise_pred * 23 - noise_list[-1] * 16 + noise_list[-2] * 5) / 12

    def predict_stage3(self, noise_pred, noise_list):
        return (
            noise_pred * 55
            - noise_list[-1] * 59
            + noise_list[-2] * 37
            - noise_list[-3] * 9
        ) / 24
