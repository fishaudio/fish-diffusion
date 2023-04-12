# Warning: This config is developing, and subject to change.

from functools import partial

from fish_diffusion.utils.dictionary import load_dictionary
from fish_diffusion.utils.pitch import get_mel_min_max, pitch_to_mel

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine.py",
    "./_base_/datasets/naive_svc.py",
]

speaker_mapping = {
    "opencpop": 0,
}

dictionary, phonemes = load_dictionary("dictionaries/opencpop.txt")

f0_min = 40.0
f0_max = 2000.0
mel_bins = 128
f0_mel_min, f0_mel_max = get_mel_min_max(f0_min, f0_max)

model = dict(
    type="PitchPredictor",
    text_encoder=dict(
        _delete_=True,
        type="FastSpeech2Encoder",
        input_size=len(phonemes),
        hidden_size=256,
    ),
    diffusion=dict(
        # timesteps=100,
        # denoiser=dict(
        #     residual_channels=384,
        # ),
        spec_min=[0],
        spec_max=[1],
    ),
    pitch_encoder=dict(
        _delete_=True,
        type="NaiveProjectionEncoder",
        input_size=128,
        output_size=256,
        use_embedding=False,
        preprocessing=partial(
            pitch_to_mel, f0_mel_min=f0_mel_min, f0_mel_max=f0_mel_max, f0_bin=mel_bins
        ),
        postprocessing=lambda x: x.squeeze(-2),
    ),
)

dataset = dict(
    _delete_=True,
    train=dict(
        type="NaiveSVSDataset",
        path="dataset/pitch-predictor/train",
        speaker_id=speaker_mapping["opencpop"],
    ),
    valid=dict(
        type="NaiveSVSDataset",
        path="dataset/pitch-predictor/valid",
        speaker_id=speaker_mapping["opencpop"],
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="OpenCpopTranscriptionToPhonemesDuration",
        phonemes=phonemes,
        transcription_path="dataset/pitch-predictor/transcriptions.txt",
    ),
    pitch_extractor=dict(
        type="ParselMouthPitchExtractor",
        keep_zeros=True,
        f0_min=f0_min,
        f0_max=f0_max,
    ),
)


dataloader = dict(
    train=dict(
        batch_size=40,
        num_workers=4,
    ),
)
