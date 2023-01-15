import os
import librosa
import numpy as np
import torch
from fish_diffusion.feature_extractors import FEATURE_EXTRACTORS
from fish_audio_preprocess.utils import separate_audio, slice_audio
from fish_diffusion.utils.audio import get_mel_from_audio
from fish_diffusion.utils.pitch import PITCH_EXTRACTORS
from fish_diffusion.utils.tensor import repeat_expand_2d
from train import DiffSVC
from mmengine import Config
from loguru import logger
import soundfile as sf
import os


@torch.no_grad()
def inference(
    config,
    checkpoint,
    input_path,
    speaker_id=0,
    pitch_adjust=0,
    device="cuda",
    extract_vocals=True,
):
    if os.path.isdir(checkpoint):
        # Find the latest checkpoint
        checkpoints = sorted(os.listdir(checkpoint))
        checkpoint = os.path.join(checkpoint, checkpoints[-1])

    audio, sr = librosa.load(input_path, sr=config.sampling_rate, mono=True)

    # Extract vocals

    if extract_vocals:
        logger.info("Extracting vocals...")
        model = separate_audio.init_model("htdemucs", device=device)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.samplerate)[None]

        # To two channels
        audio = np.concatenate([audio, audio], axis=0)
        audio = torch.from_numpy(audio).to(device)
        tracks = separate_audio.separate_audio(
            model, audio, shifts=1, num_workers=0, progress=True
        )
        audio = separate_audio.merge_tracks(tracks, filter=["vocals"]).cpu().numpy()
        audio = audio[0]
        audio = librosa.resample(audio, orig_sr=model.samplerate, target_sr=sr)

    # Slice into segments
    # TODO: 这个切片机只保留了有声音的部分, 需要修复
    segments = list(slice_audio.slice_audio(audio, sr))
    logger.info(f"Extracted {len(segments)} segments")

    # Load models
    text_features_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.text_features_extractor
    )
    pitch_extractor = PITCH_EXTRACTORS.get(config.preprocessing.pitch_extractor)
    model = DiffSVC.load_from_checkpoint(checkpoint).to(device)
    wavs = []

    assert pitch_extractor is not None, "Pitch extractor not found"

    for idx, segment in enumerate(segments):
        segment = torch.from_numpy(segment).to(device)[None]
        logger.info(
            f"Processing segment {idx + 1}/{len(segments)}, duration: {segment.shape[-1] / sr:.2f}s"
        )

        # Extract mel
        mel = get_mel_from_audio(segment, sr)

        # Extract pitch (f0)
        pitch = pitch_extractor(segment, sr, pad_to=mel.shape[-1]).float()

        # Extract text features
        text_features = text_features_extractor(segment, sr)[0]
        text_features = repeat_expand_2d(text_features, mel.shape[-1]).T

        # Predict
        src_lens = torch.tensor([mel.shape[-1]]).to(device)

        features = model.model.forward_features(
            speakers=torch.tensor([speaker_id]).long().to(device),
            contents=text_features[None].to(device),
            src_lens=src_lens,
            max_src_len=max(src_lens),
            mel_lens=src_lens,
            max_mel_len=max(src_lens),
            pitches=pitch[None].to(device),
        )

        result = model.model.diffusion.inference(features["features"])
        wav = model.vocoder.spec2wav(result[0].T, f0=pitch).cpu().numpy()

        wavs.append(wav)

    wavs = np.concatenate(wavs, axis=0)
    sf.write("output.wav", wavs, sr)

    logger.info("Done")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(
        Config.fromfile("configs/svc_hubert_soft_continuous_pitch.py"),
        "logs/diff-svc/5w6yytnv/checkpoints",
        "raw/separated/【Mia米娅】《百万个吻》Mua版-.wav",
        speaker_id=0,
        pitch_adjust=0,
        extract_vocals=False,
        device=device,
    )
