from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class IdentityEncoder(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
