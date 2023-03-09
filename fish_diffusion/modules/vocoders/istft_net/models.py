import torch
import torch.nn.functional as F
from torch.nn import Conv1d, ReflectionPad1d
from torch.nn.utils import weight_norm

from fish_diffusion.modules.vocoders.nsf_hifigan.models import LRELU_SLOPE
from fish_diffusion.modules.vocoders.nsf_hifigan.models import Generator as _Generator
from fish_diffusion.modules.vocoders.nsf_hifigan.models import init_weights


class Generator(_Generator):
    def __init__(self, h):
        super().__init__(h)

        ch = self.resblocks[-1].out_channels
        self.conv_post = weight_norm(
            Conv1d(ch, self.h.gen_istft_n_fft + 2, 7, 1, padding=3)
        )
        self.conv_post.apply(init_weights)
        self.reflection_pad = ReflectionPad1d((1, 0))

    def forward(self, x, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = self.reflection_pad(x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)

        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])

        return spec, phase
