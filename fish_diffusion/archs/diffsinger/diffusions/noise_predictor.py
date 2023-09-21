from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .uni_pc import NoiseScheduleVP, UniPC
from .uni_pc import model_wrapper as unipc_model_wrapper


def extract(a, t, n_dims=3):
    return a[t].reshape((1,) * n_dims)


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
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.dim()) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.dim()) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_start.dim()) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.dim()) * x_t
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
        a_t = extract(self.alphas_cumprod, t, x.dim())
        a_prev = extract(self.alphas_cumprod, t_prev, x.dim())
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


class UNIPCNoisePredictor(nn.Module):
    def __init__(self, betas, condition_key="conditioner"):
        super().__init__()

        # 1. Define the noise schedule.
        self.noise_schedule = NoiseScheduleVP(
            schedule="discrete", betas=torch.from_numpy(betas)
        )
        self.condition_key = condition_key

    @torch.jit.unused
    def build_tqdm(self, denoise_fn, steps):
        bar = tqdm(total=steps)

        def my_wrapper(*args, **kwargs):
            result = denoise_fn(*args, **kwargs)
            bar.update(1)
            return result

        return my_wrapper, bar

    @torch.jit.unused
    def close_tqdm(self, bar):
        bar.close()

    @torch.jit.unused  # Not support jit as of now
    def forward(
        self,
        denoise_fn,
        x,
        cond,
        progress=False,
        sampler_interval=10,
        x_masks=None,
        cond_masks=None,
    ):
        steps = self.noise_schedule.total_N // sampler_interval

        if progress:
            denoise_fn, bar = self.build_tqdm(denoise_fn, steps)

        # 2. Convert your discrete-time `model` to the continuous-time
        # noise prediction model. Here is an example for a diffusion model
        # `model` with the noise prediction type ("noise") .
        model_fn = unipc_model_wrapper(
            denoise_fn,
            self.noise_schedule,
            model_type="noise",  # or "x_start" or "v" or "score"
            model_kwargs={
                self.condition_key: cond,
                "x_masks": x_masks,
                "cond_masks": cond_masks,
            },
        )

        # 3. Define uni_pc and sample by multistep UniPC.
        # You can adjust the `steps` to balance the computation
        # costs and the sample quality.
        uni_pc = UniPC(model_fn, self.noise_schedule, variant="bh2")

        x = uni_pc.sample(
            x,
            steps=steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

        if progress:
            self.close_tqdm(bar)

        return x
