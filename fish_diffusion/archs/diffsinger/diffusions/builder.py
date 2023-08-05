from mmengine import Registry

from fish_diffusion.modules.convnext import ConvNext
from fish_diffusion.modules.wavenet import WaveNet

DIFFUSIONS = Registry("diffusions")
DENOISERS = Registry("denoisers")

DENOISERS.register_module(name="WaveNetDenoiser", module=WaveNet)
DENOISERS.register_module(name="ConvNextDenoiser", module=ConvNext)