from typing import Iterable, Optional

import torch
from loguru import logger
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,
    WhisperModel,
)
from vector_quantize_pytorch import VectorQuantize
from whisper import log_mel_spectrogram, pad_or_trim

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS


class WhisperVQ(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-medium",
        # Quantization
        codebook_dim: int = 32,
        codebook_size: int = 4096,
        codebook_decay: float = 0.9,
        threshold_ema_dead_code: int = 0,
        use_cosine_similarity: bool = True,
        downsample: bool = True,
        # Attention
        post_attention_depth: int = 2,
    ):
        super().__init__()

        self.model = WhisperModel.from_pretrained(model_name_or_path)

        # Store vars
        self.downsample = downsample
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        # Pre-quantization
        whisper_config = self.model.config
        encoder_width = whisper_config.encoder_attention_heads * 64

        self.pre_ln = nn.LayerNorm(encoder_width)
        self.pre_mlp = nn.Sequential(
            nn.Linear(encoder_width, whisper_config.encoder_ffn_dim),
            nn.GELU(),
            nn.Linear(whisper_config.encoder_ffn_dim, encoder_width),
        )

        # Quantization
        self.quantizer = VectorQuantize(
            dim=encoder_width,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            decay=codebook_decay,
            commitment_weight=1.0,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_similarity,
        )
        self.pad_embedding = nn.Parameter(torch.randn(encoder_width))

        # Post-quantization
        self.post_positional_embedding = nn.Embedding(
            whisper_config.max_source_positions, encoder_width
        )
        self.post_attention = nn.Sequential(
            *[
                WhisperEncoderLayer(
                    config=whisper_config,
                )
                for _ in range(post_attention_depth)
            ]
        )
        self.post_ln = nn.LayerNorm(encoder_width)

    def encode(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert attention_mask.ndim == 2, "Attention mask must be 2D"

            # Whisper will downsample by 2
            attention_mask = attention_mask[:, ::2]

        with torch.no_grad():
            hidden_states = self.model.encoder(
                input_features,
            ).last_hidden_state

            x = hidden_states
            if self.downsample:
                x = x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2]).mean(dim=2)

                if attention_mask is not None:
                    attention_mask = attention_mask[:, ::2]

        x = x + self.pre_mlp(self.pre_ln(x))
        quantized, indices, loss = self.quantizer(
            x, mask=attention_mask.bool() if attention_mask is not None else None
        )

        # Fill masked positions with pad embedding
        if attention_mask is not None:
            quantized[attention_mask == 0] = self.pad_embedding

        return quantized, indices, loss, hidden_states

    def decode(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Upsample
        if self.downsample:
            hidden_states = hidden_states.repeat_interleave(2, dim=1)

        # Inject position embeddings
        positions = torch.arange(
            0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
        )
        x = hidden_states + self.post_positional_embedding(positions)

        # Decode
        for layer in self.post_attention:
            x = layer(x, None, None)[0]
        hidden_states = self.post_ln(hidden_states)

        return hidden_states


@FEATURE_EXTRACTORS.register_module()
class QuantizedWhisper(BaseFeatureExtractor):
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-medium",
        quantizer_weight: str = "checkpoints/whisper-vq/vanilla/step_10000_patch.ckpt",
    ):
        super().__init__()

        self.model = WhisperVQ(model_name_or_path)

        # Restore quantizer
        errors = self.model.load_state_dict(
            torch.load(quantizer_weight, map_location="cpu"),
            strict=False,
        )
        assert (
            errors.unexpected_keys == []
        ), f"Unexpected keys: {errors.unexpected_keys}"
        assert all(
            k.startswith("model.") for k in errors.missing_keys
        ), f"Missing keys: {errors.missing_keys}"

        self.model.eval()

    @torch.no_grad()
    @torch.autocast(
        "cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.half
    )
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)
        mel = log_mel_spectrogram(audio)
        feature_len = mel.shape[1]
        mel = pad_or_trim(mel, 3000)

        mask = torch.ones(1, mel.shape[1], dtype=torch.float, device=mel.device)
        mask[:, feature_len:] = 0

        quantized, _, _, _ = self.model.encode(mel[None], mask)

        downsample = mel.shape[1] // quantized.shape[1]
        feature_len = feature_len // downsample
        features = quantized[:, :feature_len]

        return features.transpose(1, 2)


if __name__ == "__main__":
    model = QuantizedWhisper()
    print(model)

    model(torch.randn(1, 4 * 16000), 16000)
