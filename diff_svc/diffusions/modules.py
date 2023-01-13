import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pitch_tools import cwt2f0_norm, denorm_f0, f0_to_coarse
from utils.tools import dur_to_mel2ph, get_mask_from_lengths, pad

from .blocks import (
    BatchNorm1dTBC,
    Embedding,
    EncSALayer,
    LayerNorm,
    SinusoidalPositionalEmbedding,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout,
        kernel_size=None,
        num_heads=2,
        norm="ln",
        ffn_padding="SAME",
        ffn_act="gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size,
            num_heads,
            dropout=dropout,
            attention_dropout=0.0,
            relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=ffn_padding,
            norm=norm,
            act=ffn_act,
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class FFTBlocks(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        max_seq_len=2000,
        ffn_kernel_size=9,
        dropout=None,
        num_heads=2,
        use_pos_embed=True,
        use_last_norm=True,
        norm="ln",
        ffn_padding="SAME",
        ffn_act="gelu",
        use_pos_embed_alpha=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = max_seq_len
            self.padding_idx = 0
            self.pos_embed_alpha = (
                nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            )
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim,
                self.padding_idx,
                init_size=max_seq_len,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    self.hidden_size,
                    self.dropout,
                    kernel_size=ffn_kernel_size,
                    num_heads=num_heads,
                    ffn_padding=ffn_padding,
                    ffn_act=ffn_act,
                )
                for _ in range(self.num_layers)
            ]
        )
        if self.use_last_norm:
            if norm == "ln":
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == "bn":
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = (
            x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        )
        nonpadding_mask_TB = (
            1 - padding_mask.transpose(0, 1).float()[:, :, None]
        )  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = (
                layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask)
                * nonpadding_mask_TB
            )
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastspeechEncoder(FFTBlocks):
    def __init__(self, config):
        max_seq_len = config["max_seq_len"]
        hidden_size = config["transformer"]["encoder_hidden"]
        super().__init__(
            hidden_size,
            config["transformer"]["encoder_layer"],
            max_seq_len=max_seq_len * 2,
            ffn_kernel_size=config["transformer"]["ffn_kernel_size"],
            dropout=config["transformer"]["encoder_dropout"],
            num_heads=config["transformer"]["encoder_head"],
            use_pos_embed=False,  # use_pos_embed_alpha for compatibility
            ffn_padding=config["transformer"]["ffn_padding"],
            ffn_act=config["transformer"]["ffn_act"],
        )
        self.padding_idx = 0
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size,
            self.padding_idx,
            init_size=max_seq_len,
        )

        # CN Hubert large feature size: 1024
        self.proj = nn.Linear(1024, hidden_size)

    def forward(self, contents, encoder_padding_mask, spk_emb):
        """

        :param txt_tokens: [B, T]
        :param encoder_padding_mask: [B, T]
        :return: {
            "encoder_out": [T x B x C]
        }
        """

        x = self.proj(contents)
        x += spk_emb
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)

        return x

    def forward_embedding(self, txt_tokens):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
