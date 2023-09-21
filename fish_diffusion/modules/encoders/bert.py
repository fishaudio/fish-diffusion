from typing import Optional

from torch import nn
from transformers import AutoModel

from .builder import ENCODERS


@ENCODERS.register_module()
class BertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_size: Optional[int] = None,
        pretrained: bool = True,
    ):
        super(BertEncoder, self).__init__()

        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            self.bert = AutoModel.from_config(model_name)

        if output_size is None:
            output_size = self.bert.config.hidden_size

        self.proj = None
        if self.bert.config.hidden_size != output_size:
            self.proj = nn.Linear(self.bert.config.hidden_size, output_size)

        self.output_size = output_size

    def forward(self, attention_mask=None, *args, **kwargs):
        x = self.bert(*args, attention_mask=attention_mask, **kwargs).last_hidden_state

        if self.proj is not None:
            x = self.proj(x)

        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-1)

        x = x * attention_mask

        return x
