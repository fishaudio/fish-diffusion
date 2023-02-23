from typing import Iterable

from torch import nn
from torch.nn import functional as F
from whisper.model import LayerNorm, ResidualAttentionBlock, sinusoids

from fish_diffusion.denoisers.wavenet import DiffusionEmbedding

from .builder import DENOISERS


@DENOISERS.register_module()
class AttentionDenoiser(nn.Module):
    def __init__(
        self,
        mel_channels: int = 128,
        condition_channels: int = 128,
        hidden_size: int = 256,
        n_head: int = 4,
        n_layer: int = 4,
        max_len: int = 10000,
    ):
        super().__init__()

        self.mel_proj = nn.Linear(mel_channels, hidden_size)
        self.diffusion_embedding = DiffusionEmbedding(hidden_size)
        self.condition_proj = nn.Linear(condition_channels, hidden_size)

        self.register_buffer("positional_embedding", sinusoids(max_len, hidden_size))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(hidden_size, n_head) for _ in range(n_layer)]
        )

        self.ln_post = LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, mel_channels)

    def forward(self, x, diffusion_step, conditioner):
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """

        x = F.gelu(self.mel_proj(x.transpose(1, 2)))
        step = self.diffusion_embedding(diffusion_step)[:, None]
        positional_embedding = self.positional_embedding[None, : x.shape[1]]
        condition = F.gelu(self.condition_proj(conditioner.transpose(1, 2)))

        for block in self.blocks:
            x = block(x + step + condition + positional_embedding)

        x = self.ln_post(x)

        return self.output_proj(x).transpose(1, 2)
