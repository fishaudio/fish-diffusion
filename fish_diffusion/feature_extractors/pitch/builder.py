from mmengine import Registry

PITCH_EXTRACTORS = Registry("pitch_extractors")


class BasePitchExtractor:
    def __init__(self, hop_length=512, f0_min=50.0, f0_max=1100.0):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError("BasePitchExtractor is not callable.")
