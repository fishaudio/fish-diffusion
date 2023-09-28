import librosa
import torch
import torchaudio
from torch import nn


class BaseFeatureExtractor(nn.Module):
    sampling_rate = 16000

    def preprocess(self, path_or_audio, sampling_rate=None):
        if isinstance(path_or_audio, str):
            audio, sampling_rate = torchaudio.load(path_or_audio)
        else:
            audio = path_or_audio

        if len(audio.shape) > 1:
            # To mono
            audio = audio.mean(0, keepdim=True)

        if sampling_rate != self.sampling_rate:
            # There is a memory leak in torchaudio resampling
            # https://github.com/pytorch/audio/issues/2338
            audio = (
                torch.from_numpy(
                    librosa.resample(
                        audio.cpu().numpy(),
                        orig_sr=sampling_rate,
                        target_sr=self.sampling_rate,
                    )
                )
                .float()
                .to(audio.device)
            )

        return audio[0]

    @property
    def device(self):
        return next(self.parameters()).device
