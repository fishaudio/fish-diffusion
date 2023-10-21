from torch import nn
from vector_quantize_pytorch import VectorQuantize

from .builder import ENCODERS


@ENCODERS.register_module()
class MLPVectorQuantizeEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dim: int = 512,
        codebook_size: int = 4096,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = True,
    ):
        """MLP Vector Quantize Encoder

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
            dim (int, optional): Dimension. Defaults to 512.
            codebook_size (int, optional): Codebook size. Defaults to 4096.
            threshold_ema_dead_code (int, optional): Threshold for EMA dead code. Defaults to 2.
            use_cosine_sim (bool, optional): Use cosine similarity. Defaults to True.
        """

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_sim,
        )

        self.mlp_in = nn.Sequential(
            nn.Linear(input_size, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, output_size),
        )

    def forward(self, x, *args, **kwargs):
        x = self.mlp_in(x)
        quantized, _, loss = self.vq(x)
        x = self.mlp_out(quantized)

        return dict(
            features=x,
            loss=loss,
            metrics={
                "vq_loss": loss,
            },
        )
