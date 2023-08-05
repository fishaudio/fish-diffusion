import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from fish_diffusion.modules.wavenet import DiffusionEmbedding


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        diffusion_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        x = (
            x
            + self.diffusion_step_projection(diffusion_step)
            + self.condition_projection(condition)
        )

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNext(nn.Module):
    def __init__(
        self,
        mel_channels=128,
        dim=512,
        mlp_factor=4,
        condition_dim=256,
        num_layers=20,
        dilation_cycle=4,
    ):
        super(ConvNext, self).__init__()

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        )
        self.conditioner_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        )

        self.residual_layers = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=dim * mlp_factor,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        )

    def forward(self, x, diffusion_step, conditioner):
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
        condition = self.conditioner_projection(conditioner)

        for layer in self.residual_layers:
            x = layer(x, condition, diffusion_step)

        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x
