from mmengine import Registry

from fish_diffusion.modules.convnext import ConvNext, TransformerDecoderDenoiser
from fish_diffusion.modules.llama import LlamaDenoiser
from fish_diffusion.modules.wavenet import WaveNet

DIFFUSIONS = Registry("diffusions")
DENOISERS = Registry("denoisers")

DENOISERS.register_module(name="WaveNetDenoiser", module=WaveNet)
DENOISERS.register_module(name="ConvNextDenoiser", module=ConvNext)
DENOISERS.register_module(
    name="TransformerDecoderDenoiser", module=TransformerDecoderDenoiser
)
DENOISERS.register_module(name="LlamaDenoiser", module=LlamaDenoiser)
