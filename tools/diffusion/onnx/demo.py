import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from loguru import logger

from fish_diffusion.modules.pitch_extractors import ParselMouthPitchExtractor
from fish_diffusion.utils.tensor import repeat_expand


def test():
    pitch_extractor = ParselMouthPitchExtractor()
    feature_extractor = ort.InferenceSession("exported/feature_extractor.onnx")
    feature_embedding = ort.InferenceSession("exported/feature_embedding.onnx")
    diffusion = ort.InferenceSession("exported/diffusion.onnx")
    vocoder = ort.InferenceSession("checkpoints/nsf_hifigan/nsf_hifigan.onnx")
    logger.info("All models loaded.")

    audio, sr = librosa.load("raw/一半一半.wav", sr=44100, mono=True)
    mel_len = audio.shape[0] // 512

    # This can be optimized since parselmouth doesn't need torch
    audio_torch = torch.from_numpy(audio).unsqueeze(0)
    pitch = pitch_extractor(audio_torch, sr)
    pitch = pitch.numpy().astype(np.float32)
    pitch = repeat_expand(pitch, mel_len)

    logger.info(f"Pitch extracted, shape: {pitch.shape}")

    # Extract feature
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio_16k = audio_16k.astype(np.float32)
    audio_16k = np.expand_dims(audio_16k, 0)

    feature = feature_extractor.run(None, {"waveform": audio_16k})[0]
    feature = np.squeeze(feature, 0)
    feature = repeat_expand(feature.T, mel_len).T

    logger.info(f"Feature extracted, shape: {feature.shape}")

    # Embed feature
    features = feature_embedding.run(
        None,
        {
            "speakers": np.array([0], dtype=np.int64),
            "text_features": feature,
            "pitches": pitch,
        },
    )[0]

    logger.info(f"Feature embedded, shape: {features.shape}")

    # Diffusion
    mel = diffusion.run(
        None,
        {
            "condition": features,
            "sampler_interval": np.array([10], dtype=np.int64),
            "progress": np.array([False]),
        },
    )[
        0
    ]  # [1, 128, mel_len]

    logger.info(f"Mel generated, shape: {mel.shape}")

    audio = vocoder.run(
        None,
        {
            "mel": mel,
            "f0": pitch[None],
        },
    )[0]

    sf.write("generated.wav", audio[0, 0], 44100)

    logger.info("Congratulations! You have generated a speech sample! 🎉")


if __name__ == "__main__":
    test()
