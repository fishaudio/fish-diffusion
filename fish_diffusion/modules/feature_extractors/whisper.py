from typing import Iterable

import torch
from loguru import logger
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from whisper import _MODELS, log_mel_spectrogram, pad_or_trim
from whisper.model import AudioEncoder, LayerNorm, ResidualAttentionBlock, sinusoids

from .base import BaseFeatureExtractor
from .builder import FEATURE_EXTRACTORS

_PRETRAINED_MODELS = {
    "aligned-whisper-cn-25k-v1": "https://github.com/fishaudio/fish-diffusion/releases/download/v1.2b0/aligned-whisper-cn-25k-v1.ckpt",
    "aligned-whisper-cn-40k-v1.1": "https://github.com/fishaudio/fish-diffusion/releases/download/v1.2b0/aligned-whisper-cn-40k-v1.1.ckpt",
}


class PhoneEncoder(nn.Module):
    def __init__(
        self, n_phones: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.proj = nn.Embedding(n_phones, n_state, padding_idx=0)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.proj(x))
        # x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class PartialFreezedAudioEncoder(AudioEncoder):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        n_trainable_layers: int = 2,
    ):
        super().__init__(
            n_mels=n_mels, n_ctx=n_ctx, n_state=n_state, n_head=n_head, n_layer=n_layer
        )

        self.n_trainable_layers = n_trainable_layers

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the last n_trainable_layers
        for param in self.blocks[-n_trainable_layers:].parameters():
            param.requires_grad = True

        # Unfreeze the last layer norm
        for param in self.ln_post.parameters():
            param.requires_grad = True


class AlignedWhisper(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_phones: int,
        n_audio_ctx: int,
        n_audio_state: int,
        n_audio_head: int,
        n_audio_layer: int,
        n_audio_trainable_layers: int = 2,
        n_phone_state: int = 384,
        n_phone_head: int = 4,
        n_phone_layer: int = 2,
        n_outputs: int = 256,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.n_phones = n_phones

        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_audio_trainable_layers = n_audio_trainable_layers

        self.n_phone_state = n_phone_state
        self.n_phone_head = n_phone_head
        self.n_phone_layer = n_phone_layer

        self.n_outputs = n_outputs

        self.audio_encoder = PartialFreezedAudioEncoder(
            n_mels=n_mels,
            n_ctx=n_audio_ctx,
            n_state=n_audio_state,
            n_head=n_audio_head,
            n_layer=n_audio_layer,
            n_trainable_layers=n_audio_trainable_layers,
        )

        # Tiny phone encoder
        self.phone_encoder = PhoneEncoder(
            n_phones=n_phones,
            n_ctx=n_audio_ctx,
            n_state=n_phone_state,
            n_head=n_phone_head,
            n_layer=n_phone_layer,
        )

        self.audio_proj = nn.Linear(n_audio_state, n_outputs)
        self.phone_proj = nn.Linear(n_phone_state, n_outputs)

        self.phone_decoder = nn.Sequential(
            nn.Linear(n_outputs, n_outputs // 2),
            nn.Dropout(0.1),
            nn.Linear(n_outputs // 2, n_phones),
        )

    @classmethod
    def load(
        cls,
        url: str,
        n_phones: int = None,
        n_outputs: int = None,
        n_audio_trainable_layers: int = 2,
    ):
        # Load weights from the official repo
        if url in _MODELS:
            url = _MODELS[url]

        # Load weights from pretrained model
        if url in _PRETRAINED_MODELS:
            url = _PRETRAINED_MODELS[url]

        if url.startswith("http"):
            state_dict = load_state_dict_from_url(url, map_location="cpu")
        else:
            state_dict = torch.load(url, map_location="cpu")

        if n_outputs is None:
            n_outputs = state_dict["dims"].get("n_outputs", 256)

        if n_outputs is None:
            raise ValueError(
                "n_outputs must be provided if not found in the model state dict (probably loaded from OpenAI's model hub)."
            )

        if "n_phones" not in state_dict["dims"]:
            state_dict["dims"]["n_phones"] = n_phones

        if "n_audio_trainable_layers" not in state_dict["dims"]:
            state_dict["dims"]["n_audio_trainable_layers"] = n_audio_trainable_layers

        model = cls(
            n_mels=state_dict["dims"]["n_mels"],
            n_audio_ctx=state_dict["dims"]["n_audio_ctx"],
            n_audio_state=state_dict["dims"]["n_audio_state"],
            n_audio_head=state_dict["dims"]["n_audio_head"],
            n_audio_layer=state_dict["dims"]["n_audio_layer"],
            n_audio_trainable_layers=state_dict["dims"]["n_audio_trainable_layers"],
            n_phones=state_dict["dims"]["n_phones"],
            n_outputs=n_outputs,
        )

        model_state_dict = {}
        for k, v in state_dict["model_state_dict"].items():
            if k.startswith("encoder."):
                model_state_dict[f"audio_{k}"] = v
            elif (
                k.startswith("phone_encoder.")
                or k.startswith("phone_proj.")
                or k.startswith("phone_decoder.")
                or k.startswith("audio_encoder.")
                or k.startswith("audio_proj.")
            ):
                model_state_dict[k] = v

        results = model.load_state_dict(model_state_dict, strict=False)
        for i in results.missing_keys:
            if i.startswith("audio_encoder."):
                raise ValueError(
                    f"Mismatch between the model and the provided state dict: {i} is missing."
                )

        assert results.unexpected_keys == []

        return model

    def save(self, path: str):
        state_dict = {"model_state_dict": self.state_dict()}

        state_dict["dims"] = dict(
            n_mels=self.n_mels,
            n_audio_ctx=self.n_audio_ctx,
            n_audio_state=self.n_audio_state,
            n_audio_head=self.n_audio_head,
            n_audio_layer=self.n_audio_layer,
            n_audio_trainable_layers=self.n_audio_trainable_layers,
            n_phones=self.n_phones,
            n_phone_state=self.n_phone_state,
            n_phone_head=self.n_phone_head,
            n_phone_layer=self.n_phone_layer,
            n_outputs=self.n_outputs,
        )

        torch.save(state_dict, path)

    def forward_audio(self, x: Tensor):
        x = self.audio_encoder(x)
        x = self.audio_proj(x)

        return x

    def forward_phones(self, x: Tensor):
        x = self.phone_encoder(x)
        x = self.phone_proj(x)

        return x

    def forward_decoder(self, x: Tensor):
        x = self.phone_decoder(x)

        return x


@FEATURE_EXTRACTORS.register_module()
class AlignedWhisperForAudio(BaseFeatureExtractor):
    def __init__(
        self,
        checkpoint: str = "aligned-whisper-cn-40k-v1.1",
        checkpoint_path: str = None,
    ):
        super().__init__()

        if checkpoint_path is not None:
            checkpoint = checkpoint_path

            logger.warning(
                "The `checkpoint_path` argument is deprecated and will be removed in a future release. "
                "Please use `checkpoint` instead."
            )

        self.model = AlignedWhisper.load(checkpoint)
        self.model.eval()

    @torch.no_grad()
    def forward(self, path_or_audio, sampling_rate=None):
        audio = self.preprocess(path_or_audio, sampling_rate)
        mel = log_mel_spectrogram(audio)
        feature_len = mel.shape[1] // 2
        mel = pad_or_trim(mel, 3000)

        features = self.model.forward_audio(mel[None])
        features = features[:, :feature_len]

        return features.transpose(1, 2)


@FEATURE_EXTRACTORS.register_module()
class AlignedWhisperForPhones(BaseFeatureExtractor):
    def __init__(
        self,
        checkpoint: str = "aligned-whisper-cn-40k-v1.1",
        checkpoint_path: str = None,
    ):
        super().__init__()

        if checkpoint_path is not None:
            checkpoint = checkpoint_path

            logger.warning(
                "The `checkpoint_path` argument is deprecated and will be removed in a future release. "
                "Please use `checkpoint` instead."
            )

        self.model = AlignedWhisper.load(checkpoint)
        self.model.eval()

    @torch.no_grad()
    def forward(self, phones: Tensor):
        phones_len = phones.shape[-1]
        phones = pad_or_trim(phones, 1500)
        features = self.model.forward_phones(phones[None])
        features = features[:, :phones_len]

        return features.transpose(1, 2)
