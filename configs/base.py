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

_base_ = [
    "./_base_/trainers/base.py",
    "./_base_/datasets/audio_folder.py",
]

sampling_rate = 44100
mel_channels = 128
hidden_size = 256

model = dict(
    diffusion=dict(
        type="GaussianDiffusion",
        mel_channels=mel_channels,
        keep_bins=128,
        noise_schedule="linear",
        timesteps=1000,
        max_beta=0.01,
        s=0.008,
        noise_loss="smoothed-l1",
        denoiser=dict(
            type="WaveNetDenoiser",
            mel_channels=mel_channels,
            d_encoder=hidden_size,
            residual_channels=512,
            residual_layers=20,
            dropout=0.2,
        ),
    ),
    text_encoder=dict(
        type="FastSpeech2Encoder",
        input_size=392,  # IPA symbols
        max_seq_len=4096,
        num_layers=4,
        hidden_size=256,
        ffn_kernel_size=9,
        dropout=0.1,
        num_heads=2,
    ),
    speaker_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=1,
        output_size=hidden_size,
        use_embedding=True,
    ),
    pitch_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=256,
        output_size=hidden_size,
        use_embedding=True,
    ),
    vocoder=dict(
        type="NsfHifiGAN",
        checkpoint_path="checkpoints/nsf_hifigan/model",
        sampling_rate=sampling_rate,
        mel_channels=mel_channels,
    ),
)
