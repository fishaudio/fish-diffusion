"""
 Copyright 2023 Lengyue

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


sampling_rate = 44100
mel_channels = 128
hidden_size = 256

model = dict(
    diffusion=dict(
        type="GaussianDiffusion",
        noise_schedule=dict(
            type="linear",
            timesteps=1000,
            max_beta=0.01,
        ),
        noise_loss="l2",
        spec_stat_file="dataset/stats.json",
    ),
    denoiser=dict(
        type="WaveNet",
        n_mel_channels=mel_channels,
        d_encoder=hidden_size,
        residual_channels=512,
        residual_layers=20,
        dropout=0.2,
    ),
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
        sampling_rate=sampling_rate,
        mel_channels=mel_channels,
    ),
)

data = dict()
