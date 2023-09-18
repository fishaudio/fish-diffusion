from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super(TransformerEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        if input_size != hidden_size:
            self.in_proj = nn.Linear(input_size, hidden_size)

        if output_size != hidden_size:
            self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_mask):
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(-1)

        if hasattr(self, "in_proj"):
            x = self.in_proj(x * x_mask)

        x = self.encoder(x * x_mask, src_key_padding_mask=(~x_mask.bool()).squeeze(-1))

        if hasattr(self, "out_proj"):
            x = self.out_proj(x * x_mask)

        return x * x_mask
