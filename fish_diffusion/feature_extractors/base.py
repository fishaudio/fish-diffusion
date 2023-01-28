import librosa
import torchaudio
from torch import nn


class BaseFeatureExtractor(nn.Module):
    def preprocess(self, path_or_audio, sampling_rate=None):
        if isinstance(path_or_audio, str):
            audio, sampling_rate = torchaudio.load(path_or_audio)
        else:
            audio = path_or_audio

        if len(audio.shape) > 1:
            # To mono
            audio = audio.mean(0, keepdim=True)

        if sampling_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sampling_rate, new_freq=16000
            )

        return audio[0]

    @property
    def device(self):
        return next(self.parameters()).device
