from typing import List, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaConfig, LlamaModel

from fish_diffusion.modules.wavenet import DiffusionEmbedding


class LlamaDenosierConfig(LlamaConfig):
    def __init__(
        self,
        *args,
        diffusion_channels: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.diffusion_channels = diffusion_channels


class LlamaDenoiser(LlamaModel):
    def __init__(self, **config):
        if isinstance(config, LlamaDenosierConfig) is False:
            config = LlamaDenosierConfig(**config)

        super().__init__(config)

        self.diffusion_embedding = DiffusionEmbedding(config.hidden_size)
        self.mel_len_embedding = nn.Linear(1, config.hidden_size)
        self.in_proj = nn.Linear(config.diffusion_channels, config.hidden_size)
        self.merge_proj = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.diffusion_channels)

    def forward(self, x, diffusion_step, conditioner, x_masks=None, cond_masks=None):
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, E] tokens
        :param x_masks: [B, T] float mask
        :param cond_masks: [B, E] float mask
        :return:
        """

        x = x.transpose(1, 2)

        if x_masks is None:
            x_masks = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        elif x_masks.dtype == torch.bool:
            x_masks = (~x_masks).float()

        if conditioner.ndim == 3:
            assert conditioner.shape[-1] == 1
            conditioner = conditioner.squeeze(-1)

        if cond_masks is None:
            cond_masks = torch.ones_like(conditioner, device=x.device, dtype=x.dtype)
        elif cond_masks.dtype == torch.bool:
            cond_masks = (~cond_masks).float()

        inputs_embeds = self.embed_tokens(conditioner)
        attention_mask = torch.concatenate([x_masks, cond_masks], dim=1)

        # Diffusion embedding
        diffusion_embeds = self.diffusion_embedding(diffusion_step)
        if diffusion_embeds.ndim != 3:
            diffusion_embeds = diffusion_embeds[:, None]

        if diffusion_step.shape[0] != x.shape[0]:
            diffusion_embeds = diffusion_embeds.repeat(x.shape[0], 1, 1)

        diffusion_embeds = diffusion_embeds.repeat(1, x.shape[1], 1)

        # Mel length embedding
        mel_lens = x_masks.sum(dim=1, keepdim=True).float()
        mel_lens = torch.log(mel_lens)[:, None]
        mel_lens_embeds = self.mel_len_embedding(mel_lens)
        mel_lens_embeds = mel_lens_embeds.repeat(1, x.shape[1], 1)

        # Merge
        x = self.in_proj(x)
        x = torch.concatenate([diffusion_embeds, mel_lens_embeds, x], dim=2)
        x = self.merge_proj(x)

        # Control tokens concat ((diffusion + mel length) embedding + x)
        inputs_embeds = torch.concatenate([inputs_embeds, x], dim=1)

        outputs = (
            super()
            .forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            )
            .last_hidden_state
        )

        # Extract x
        x = outputs[:, -diffusion_embeds.shape[1] :, :]
        x = self.out_proj(x)

        return x.transpose(1, 2)


if __name__ == "__main__":
    # Print trainable parameters
    model = LlamaDenoiser(
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=768 * 4,
        max_position_embeddings=4096,
        model_type="llama",
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        tie_word_embeddings=False,
        vocab_size=32000,
    )
    pms = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("Trainable parameters:")
    # for p in pms:
    #     print(p.shape)
    print("Total:", sum([p.numel() for p in pms]) / 1_000_000, "M")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(tokenizer.decode(tokenizer.encode("[spk] Aria [txt] 真的难绷 [mel]")))
    print(tokenizer.bos_token)
    print(tokenizer.eos_token)

    x = torch.randn(1, 128, 100)
    diffusion_step = torch.randint(1, 100, (1, 1))
    condition = torch.randint(1, 200, (1, 10))
    x = model(x, diffusion_step, condition)
    print(x.shape)
