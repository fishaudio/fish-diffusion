from typing import Iterable

from torch import Tensor, nn
from torch.nn import functional as F
from whisper.model import AudioEncoder, LayerNorm, ResidualAttentionBlock, sinusoids

from .builder import ENCODERS


@ENCODERS.register_module()
class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_head: int = 4,
        n_layer: int = 4,
        max_len: int = 10000,
    ):
        super().__init__()

        self.proj = nn.Linear(input_size, output_size)
        self.register_buffer("positional_embedding", sinusoids(max_len, output_size))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(output_size, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(output_size)

    def forward(self, x: Tensor, encoder_padding_mask: Tensor = None):
        x = F.gelu(self.proj(x))

        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)

        return x
