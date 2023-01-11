import librosa
from torch import nn


class BaseFeatureExtractor(nn.Module):
    def preprocess(self, path_or_audio, sampling_rate=None):
        if isinstance(path_or_audio, str):
            audio, sampling_rate = librosa.load(path_or_audio)
        else:
            audio = path_or_audio

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))

        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

        return audio

    @property
    def device(self):
        return next(self.parameters()).device
