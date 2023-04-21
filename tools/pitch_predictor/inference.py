import json
from copy import deepcopy

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mmengine import Config

from fish_diffusion.archs.diffsinger.diffsinger import DiffSingerLightning
from fish_diffusion.modules.pitch_extractors.builder import BasePitchExtractor
from fish_diffusion.modules.pitch_extractors.crepe import MaskedAvgPool1d
from fish_diffusion.utils.dictionary import load_dictionary
from fish_diffusion.utils.inference import load_checkpoint
from fish_diffusion.utils.pitch import (
    get_mel_min_max,
    mel_scale_to_pitch,
    pitch_to_mel_scale,
    viterbi_decode,
)
from fish_diffusion.utils.tensor import repeat_expand

# plt.imshow(mel_scale.cpu().numpy().T)
# plt.savefig("mel_scale.png")
# plt.plot(predicted_pitches.cpu().numpy())
# plt.plot(f0.cpu().numpy())
# plt.savefig("pitches.png")
# exit()

device = "cuda:1"
config = Config.fromfile("configs/pitch_predictor.py")
# config.model.diffusion.sampler_interval = 1
model = load_checkpoint(
    config,
    "logs/PitchPredictor/version_None/checkpoints/epoch=851-step=40000-valid_loss=0.08.ckpt",
    device="cpu",
    model_cls=DiffSingerLightning,
)
model.eval()
model.to(device)

ds_file = "自动音高测试.ds"
data = json.load(open(ds_file))

dictionary, phonemes = load_dictionary("dictionaries/opencpop.txt")
dictionary_extension, phonemes_extension = load_dictionary(
    "dictionaries/opencpop-extension.txt"
)
dictionary["AP"] = ["AP"]
dictionary["SP"] = ["SP"]

dictionary_extension_greedy = []
for key, value in dictionary_extension.items():
    dictionary_extension_greedy.append((key, value))
dictionary_extension_greedy.sort(key=lambda x: len(x[1]), reverse=True)
dictionary_extension_greedy.extend([("SP", ["SP"]), ("AP", ["AP"])])


def convert_extension_phones_to_words(
    phones: list[str], phones_dur: list[float]
) -> tuple[list[str], list[list[float]]]:
    def backtrack(
        remaining_phones,
        remaining_phones_dur,
        current_words,
        current_words_dur,
        position,
    ):
        if not remaining_phones:
            return current_words, current_words_dur

        for i in range(position, len(dictionary_extension_greedy)):
            key, value = dictionary_extension_greedy[i]

            if remaining_phones[: len(value)] == value:
                words = current_words + [key]
                words_dur = current_words_dur + [remaining_phones_dur[: len(value)]]
                new_phones = remaining_phones[len(value) :]
                new_phones_dur = remaining_phones_dur[len(value) :]
                result = backtrack(new_phones, new_phones_dur, words, words_dur, 0)

                if result:
                    return result

        return None

    return backtrack(phones, phones_dur, [], [], 0)


def convert_words_to_normal(
    phones: list[str], phones_dur: list[list[float]]
) -> tuple[list[str], list[float]]:
    new_phones = []
    new_phones_dur = []

    for phone, phone_dur in zip(phones, phones_dur):
        temp = dictionary[phone]
        new_phones.extend(temp)

        if len(temp) == len(phone_dur):
            new_phones_dur.extend(phone_dur)
        elif len(temp) == 1 and len(phone_dur) == 2:
            new_phones_dur.extend(sum(phone_dur))
        else:
            print(phone, phone_dur)
            raise ValueError("Invalid phone duration")

    return new_phones, new_phones_dur


new_segments = []
for segment in data:
    f0_seq = list(map(float, segment["f0_seq"].split(" ")))
    f0_timestep = float(segment["f0_timestep"])

    note_seq = segment["note_seq"].split(" ")
    note_dur_seq = list(map(float, segment["note_dur_seq"].split(" ")))

    phones_seq = segment["ph_seq"].split(" ")
    phones_dur_seq = list(map(float, segment["ph_dur"].split(" ")))

    # assert len(note_seq) == len(note_dur_seq)
    # assert note_dur_seq == phones_dur_seq

    # Conver from extention phonemes to normal phonemes
    # words_seq, words_dur_seq = convert_extension_phones_to_words(
    #     phones_seq, phones_dur_seq
    # )
    # phones_seq, phones_dur_seq = convert_words_to_normal(words_seq, words_dur_seq)

    assert len(phones_seq) == len(phones_dur_seq)

    mel_len = len(f0_seq)

    cumsum_durations = np.cumsum(phones_dur_seq)
    alignment_factor = mel_len / cumsum_durations[-1]

    # Create one-hot encoding for phonemes
    text_features = F.one_hot(
        torch.tensor([config.phonemes.index(i) for i in phones_seq]),
        num_classes=len(config.phonemes),
    ).float()

    # Create phones to mel alignment
    phones2mel = torch.zeros(mel_len, dtype=torch.long)

    for i, sum_duration in enumerate(cumsum_durations):
        current_idx = int(sum_duration * alignment_factor)
        previous_idx = int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
        phones2mel[previous_idx:current_idx] = i

    f0s = []
    for note in note_seq:
        if note == "rest":
            f0s.append(0)
            continue

        if "/" in note:
            note, _ = note.split("/")

        f0 = librosa.note_to_hz(note)
        f0s.append(f0)

    f0 = torch.tensor(f0s, dtype=torch.float)

    # create note to mel alignment
    cumsum_durations = np.cumsum(note_dur_seq)
    alignment_factor = mel_len / cumsum_durations[-1]
    notes2mel = torch.zeros(mel_len, dtype=torch.long)

    for i, sum_duration in enumerate(cumsum_durations):
        current_idx = int(sum_duration * alignment_factor)
        previous_idx = int(cumsum_durations[i - 1] * alignment_factor) if i > 0 else 0
        notes2mel[previous_idx:current_idx] = i

    pitches = torch.gather(f0, 0, notes2mel)
    # pitches *= 2 ** (12 / 12)

    # Predict pitch
    pitch_shift = torch.zeros((1, 1), device=device)
    speakers = torch.zeros((1, 1), device=device)
    contents_lens = torch.tensor([text_features.shape[0]], device=device)
    mel_lens = torch.tensor([mel_len], device=device)
    # print(pitch_shift, speakers, contents_lens, mel_lens)
    # exit()

    with torch.no_grad():
        features = model.model.forward_features(
            speakers=speakers.to(device),
            contents=text_features[None].to(device),
            contents_lens=contents_lens,
            contents_max_len=max(contents_lens),
            mel_lens=mel_lens,
            mel_max_len=max(mel_lens),
            pitches=pitches[None].to(device),
            # pitches=torch.tensor([f0_seq], dtype=torch.float, device=device),
            pitch_shift=pitch_shift,
            phones2mel=phones2mel[None].to(device),
        )

        result = model.model.diffusion(features["features"], progress=True)[0]

    # Save result
    weights = (result * 10).softmax(axis=1)
    decoded = viterbi_decode(weights.T)
    # argmax = weights.argmax(axis=1)
    weights = weights[torch.arange(result.shape[0]), decoded]

    # Mel to pitch
    f0_mel_min, f0_mel_max = config.f0_mel_min, config.f0_mel_max
    f0 = mel_scale_to_pitch(decoded, f0_mel_min, f0_mel_max, 128)
    # f0[weights < 0.1] = torch.nan

    filter_kernal = 5
    mean_filter = MaskedAvgPool1d(filter_kernal, 1, padding=filter_kernal // 2)
    f0 = mean_filter(f0[None])[0]
    # f0[weights < 0.1] = 0

    # Interpolate
    extractor = BasePitchExtractor(
        f0_min=config.preprocessing.pitch_extractor.f0_min,
        f0_max=config.preprocessing.pitch_extractor.f0_max,
        keep_zeros=False,
    )
    raw_f0_len = len(f0_seq)
    # if raw_f0_len == len(f0):
    #     print("No interpolation needed")
    # else:
    f0 = extractor.post_process(f0, 512, f0, raw_f0_len)

    # f0 *= 2 ** (-6 / 12)
    f0 = f0.cpu().detach().numpy()

    from matplotlib import pyplot as plt

    plt.cla()
    plt.imshow(
        (result * 10).softmax(axis=1).cpu().detach().numpy().T,
        aspect="auto",
        origin="lower",
    )
    # downsample_f0 = repeat_expand(torch.tensor(f0_seq), result.shape[0])
    plt.plot(
        pitch_to_mel_scale(torch.tensor(f0_seq), f0_mel_min, f0_mel_max, 128),
        color="red",
        alpha=1,
    )
    # xx = pitch_to_mel_scale(torch.tensor(f0_seq), f0_mel_min, f0_mel_max, 128)
    # rv = mel_scale_to_pitch(xx, f0_mel_min, f0_mel_max, 128)
    plt.plot(
        pitch_to_mel_scale(torch.from_numpy(f0), f0_mel_min, f0_mel_max, 128),
        color="yellow",
        alpha=1,
    )
    plt.plot(pitch_to_mel_scale(pitches, f0_mel_min, f0_mel_max, 128), color="green")
    plt.savefig("mel.png")

    plt.figure()
    # plt.plot(f0_seq)
    plt.plot(f0)
    # plt.plot(pitches.cpu().detach().numpy())
    plt.ylim(40, 1600)
    plt.savefig("f0.png")

    new_segment = deepcopy(segment)
    new_segment["f0_seq"] = " ".join([f"{i:.1f}" for i in f0])
    new_segments.append(new_segment)

    # input("Press enter to continue...")

with open("test.ds", "w") as f:
    json.dump(new_segments, f)
