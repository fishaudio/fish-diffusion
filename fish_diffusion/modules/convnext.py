import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
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
        x_masks: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        if diffusion_step is not None:
            x = x + self.diffusion_step_projection(diffusion_step)

        if condition is not None:
            if cond_masks is not None:
                condition = condition.masked_fill(cond_masks[:, None, :], 0.0)

            x = x + self.condition_projection(condition)

        if x_masks is not None:
            x = x.masked_fill(x_masks[:, None, :], 0.0)

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

        if x_masks is not None:
            x = x.masked_fill(x_masks[:, None, :], 0.0)

        return x


class CrossAttentionBlock(nn.TransformerDecoderLayer):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        nhead: int = 8,
    ):
        super().__init__(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=intermediate_dim,
            activation="gelu",
            batch_first=True,
        )

        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.register_buffer("positional_embedding", self.get_embedding(dim))
        self.position_scale_query = nn.Parameter(torch.ones(1))
        self.position_scale_key = nn.Parameter(torch.ones(1))

    def get_embedding(self, embedding_dim, num_embeddings=4096):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )

        return emb

    def forward(self, x, condition, diffusion_step, x_masks=None, cond_masks=None):
        if diffusion_step is not None:
            x = x + self.diffusion_step_projection(diffusion_step)

        # Apply positional encoding to both x and condition
        x = x.transpose(1, 2)
        condition = condition.transpose(1, 2)

        # self.get_embedding(dim)
        x = x + self.positional_embedding[: x.size(1)][None] * self.position_scale_query
        condition = (
            condition
            + self.positional_embedding[: condition.size(1)][None]
            * self.position_scale_key
        )

        return (
            super()
            .forward(
                tgt=x,
                memory=condition,
                tgt_key_padding_mask=x_masks,
                memory_key_padding_mask=cond_masks,
            )
            .transpose(1, 2)
        )


class ConvNext(nn.Module):
    def __init__(
        self,
        mel_channels=128,
        dim=512,
        mlp_factor=4,
        condition_dim=256,
        num_layers=20,
        dilation_cycle=4,
        gradient_checkpointing=False,
        cross_attention=False,
        cross_every_n_layers=5,
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

        self.residual_layers = nn.ModuleList([])

        for i in range(num_layers):
            if cross_attention and i % cross_every_n_layers == 0:
                self.residual_layers.append(
                    CrossAttentionBlock(
                        dim=dim,
                        intermediate_dim=dim * mlp_factor,
                    )
                )

            self.residual_layers.append(
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=dim * mlp_factor,
                    dilation=2 ** (i % dilation_cycle),
                )
            )

        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        )

        self.gradient_checkpointing = gradient_checkpointing
        self.cross_attention = cross_attention

    def forward(self, x, diffusion_step, conditioner, x_masks=None, cond_masks=None):
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, E]
        :param x_masks: [B, T] bool mask
        :param cond_masks: [B, E] bool mask
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

        if x_masks is not None:
            x = x.masked_fill(x_masks[:, None, :], 0.0)

        if cond_masks is not None:
            condition = condition.masked_fill(cond_masks[:, None, :], 0.0)

        for layer in self.residual_layers:
            is_cross_layer = isinstance(layer, CrossAttentionBlock)
            temp_condition = (
                condition
                if ((self.cross_attention is False) or is_cross_layer)
                else None
            )

            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, temp_condition, diffusion_step, x_masks, cond_masks
                )
            else:
                x = layer(x, temp_condition, diffusion_step, x_masks, cond_masks)

        x = self.output_projection(x)  # [B, 128, T]
        if x_masks is not None:
            x = x.masked_fill(x_masks[:, None, :], 0.0)

        return x[:, None] if use_4_dim else x


class TransformerDecoderDenoiser(nn.Module):
    def __init__(
        self,
        mel_channels=128,
        dim=512,
        mlp_factor=4,
        condition_dim=256,
        num_layers=12,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(mel_channels, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        )
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        )

        self.register_buffer("positional_embedding", self.get_embedding(dim))
        self.position_scale_query = nn.Parameter(torch.ones(1))
        self.position_scale_key = nn.Parameter(torch.ones(1))

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=dim,
                    nhead=8,
                    dim_feedforward=dim * mlp_factor,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        )

        self.gradient_checkpointing = gradient_checkpointing

    def get_embedding(self, embedding_dim, num_embeddings=4096):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )

        return emb

    def forward(self, x, diffusion_step, conditioner, x_masks=None, cond_masks=None):
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, E]
        :param x_masks: [B, T] bool mask
        :param cond_masks: [B, E] bool mask
        :return:
        """

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x).transpose(1, 2)  # x [B, T, residual_channel]
        x_pos = self.positional_embedding[None, : x.size(1)] * self.position_scale_query
        x = x + x_pos

        condition = self.condition_projection(conditioner).transpose(1, 2)
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(1)
        condition_pos = (
            self.positional_embedding[None, : condition.size(1)]
            * self.position_scale_key
        )
        condition = condition + condition_pos + diffusion_step

        if x_masks is not None:
            x = x.masked_fill(x_masks[..., None], 0.0)

        if cond_masks is not None:
            condition = condition.masked_fill(cond_masks[..., None], 0.0)

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    condition,
                    x_masks,
                    cond_masks,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    tgt=x,
                    memory=condition,
                    tgt_key_padding_mask=x_masks,
                    memory_key_padding_mask=cond_masks,
                )

        x = x.transpose(1, 2)
        x = self.output_projection(x)  # [B, 128, T]
        if x_masks is not None:
            x = x.masked_fill(x_masks[:, None], 0.0)

        return x


if __name__ == "__main__":
    import torch

    gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model = TransformerDecoderDenoiser().cuda()
    x = torch.randn(8, 128, 1024).cuda()
    diffusion_step = torch.randint(0, 1000, (8,)).cuda()
    conditioner = torch.randn(8, 256, 256).cuda()
    x_masks = torch.randint(0, 2, (8, 1024)).bool().cuda()
    cond_masks = torch.randint(0, 2, (8, 256)).bool().cuda()
    y = model(x, diffusion_step, conditioner, x_masks, cond_masks)
    print(y.shape)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3 - gpu_memory_usage
    print(f"GPU memory usage: {gpu_memory_usage:.2f} GB")
