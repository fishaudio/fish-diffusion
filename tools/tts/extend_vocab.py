import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# pick the model type
model_type = "baichuan-inc/Baichuan2-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(
    model_type, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(model_type, trust_remote_code=True)


def _get_resized_lm_head(
    old_lm_head, new_num_tokens=None, transposed=None
) -> nn.Linear:
    new_weights = torch.empty(
        (new_num_tokens - old_lm_head.weight.shape[0], old_lm_head.weight.shape[1])
    )
    nn.init.kaiming_normal_(new_weights, a=math.sqrt(5))

    old_lm_head.weight.data = torch.cat((old_lm_head.weight.data, new_weights), dim=0)

    return old_lm_head


model._get_resized_lm_head = _get_resized_lm_head

# new tokens
new_tokens = [f"<encodec_{i}>" for i in range(1024)]

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(new_tokens)

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# Try tokenizing a new sequence
sequence = "Test <encodec_0><encodec_1023>"
encoded = tokenizer.encode(sequence)
print(encoded)

model.save_pretrained("./checkpoints/baichuan2-7b-base-extend")
tokenizer.save_pretrained("./checkpoints/baichuan2-7b-base-extend")

tokenizer.push_to_hub("fishaudio/baichuan2-7b-extend")
model.push_to_hub("fishaudio/baichuan2-7b-extend")
