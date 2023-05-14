"""
This file is copied from https://github.com/openvpi/DiffSinger/blob/refactor/pipelines/no_midi_preparation.ipynb

It aims to reduce errors for long utterances and detcect aspiration which is not detected by MFA.
"""


import librosa
import numpy as np
import parselmouth as pm
import textgrid as tg
import tqdm
from fish_audio_preprocess.utils.file import list_files

# Load dictionary
dict_path = "dictionaries/english-arpa.txt"
with open(dict_path, "r", encoding="utf8") as f:
    rules = [ln.strip().split("\t") for ln in f.readlines()]

dictionary = {}
phoneme_set = set()
for r in rules:
    phonemes = r[1].split()

    dictionary[r[0]] = phonemes
    phoneme_set.update(phonemes)

# Edge case
dictionary["n"] = ["n"]

########################################

# Configuration for voice arguments based on your dataset
f0_min = 40.0  # Minimum value of pitch
f0_max = 1100.0  # Maximum value of pitch
br_len = 0.1  # Minimum length of aspiration in seconds
br_db = -60.0  # Threshold of RMS in dB for detecting aspiration
br_centroid = 2000.0  # Threshold of spectral centroid in Hz for detecting aspiration

# Other arguments, do not edit unless you understand them
time_step = 0.005  # Time step for feature extraction
min_space = 0.04  # Minimum length of space in seconds
voicing_thresh_vowel = 0.45  # Threshold of voicing for fixing long utterances
voicing_thresh_breath = 0.6  # Threshold of voicing for detecting aspiration
br_win_sz = 0.05  # Size of sliding window in seconds for detecting aspiration

########################################

# import utils.tg_optimizer as optimizer

for file in tqdm.tqdm(
    list_files("dataset/mfa-data/english/LJSpeech/normed", ".wav", recursive=True)
):
    try:
        textgrid = tg.TextGrid()
        textgrid.read(str(file.with_suffix(".TextGrid")))
        words = textgrid[0]
        phones = textgrid[1]
        sound = pm.Sound(str(file))
        f0_voicing_breath = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=voicing_thresh_breath,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        ).selected_array["frequency"]
        f0_voicing_vowel = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=voicing_thresh_vowel,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        ).selected_array["frequency"]
        y, sr = librosa.load(str(file), sr=24000, mono=True)
        hop_size = int(time_step * sr)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=2048, hop_length=hop_size
        ).squeeze(0)

        # Fix long utterances
        i = j = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != "":
                i += 1
                j += len(dictionary[word.mark])
                continue
            if i == 0:
                i += 1
                j += 1
                continue
            prev_word = words[i - 1]
            prev_phone = phones[j - 1]
            # Extend length of long utterances
            while word.minTime < word.maxTime - time_step:
                pos = min(f0_voicing_vowel.shape[0] - 1, int(word.minTime / time_step))
                if f0_voicing_vowel[pos] < f0_min:
                    break
                prev_word.maxTime += time_step
                prev_phone.maxTime += time_step
                word.minTime += time_step
                phone.minTime += time_step
            i += 1
            j += 1

        # Detect aspiration
        i = j = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != "":
                i += 1
                j += len(dictionary[word.mark])
                continue
            if word.maxTime - word.minTime < br_len:
                i += 1
                j += 1
                continue
            ap_ranges = []
            br_start = None
            win_pos = word.minTime
            while win_pos + br_win_sz <= word.maxTime:
                all_noisy = (
                    f0_voicing_breath[
                        int(win_pos / time_step) : int(
                            (win_pos + br_win_sz) / time_step
                        )
                    ]
                    < f0_min
                ).all()
                rms_db = 20 * np.log10(
                    np.clip(
                        sound.get_rms(from_time=win_pos, to_time=win_pos + br_win_sz),
                        a_min=1e-12,
                        a_max=1,
                    )
                )
                # print(win_pos, win_pos + br_win_sz, all_noisy, rms_db)
                if all_noisy and rms_db >= br_db:
                    if br_start is None:
                        br_start = win_pos
                else:
                    if br_start is not None:
                        br_end = win_pos + br_win_sz - time_step
                        if br_end - br_start >= br_len:
                            centroid = spectral_centroid[
                                int(br_start / time_step) : int(br_end / time_step)
                            ].mean()
                            if centroid >= br_centroid:
                                ap_ranges.append((br_start, br_end))
                        br_start = None
                        win_pos = br_end
                win_pos += time_step
            if br_start is not None:
                br_end = win_pos + br_win_sz - time_step
                if br_end - br_start >= br_len:
                    centroid = spectral_centroid[
                        int(br_start / time_step) : int(br_end / time_step)
                    ].mean()
                    if centroid >= br_centroid:
                        ap_ranges.append((br_start, br_end))
            # print(ap_ranges)
            if len(ap_ranges) == 0:
                i += 1
                j += 1
                continue
            words.removeInterval(word)
            phones.removeInterval(phone)
            if word.minTime < ap_ranges[0][0]:
                words.add(minTime=word.minTime, maxTime=ap_ranges[0][0], mark=None)
                phones.add(minTime=phone.minTime, maxTime=ap_ranges[0][0], mark=None)
                i += 1
                j += 1
            for k, ap in enumerate(ap_ranges):
                if k > 0:
                    words.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                    phones.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                    i += 1
                    j += 1
                words.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark="AP")
                phones.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark="AP")
                i += 1
                j += 1
            if ap_ranges[-1][1] < word.maxTime:
                words.add(minTime=ap_ranges[-1][1], maxTime=word.maxTime, mark=None)
                phones.add(minTime=ap_ranges[-1][1], maxTime=phone.maxTime, mark=None)
                i += 1
                j += 1

        # Remove short spaces
        i = j = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != "":
                i += 1
                j += 1 if word.mark == "AP" else len(dictionary[word.mark])
                continue
            if word.maxTime - word.minTime >= min_space:
                word.mark = "SP"
                phone.mark = "SP"
                i += 1
                j += 1
                continue
            if i == 0:
                if len(words) >= 2:
                    words[i + 1].minTime = word.minTime
                    phones[j + 1].minTime = phone.minTime
                    words.removeInterval(word)
                    phones.removeInterval(phone)
                else:
                    break
            elif i == len(words) - 1:
                if len(words) >= 2:
                    words[i - 1].maxTime = word.maxTime
                    phones[j - 1].maxTime = phone.maxTime
                    words.removeInterval(word)
                    phones.removeInterval(phone)
                else:
                    break
            else:
                words[i - 1].maxTime = words[i + 1].minTime = (
                    word.minTime + word.maxTime
                ) / 2
                phones[j - 1].maxTime = phones[j + 1].minTime = (
                    phone.minTime + phone.maxTime
                ) / 2
                words.removeInterval(word)
                phones.removeInterval(phone)

        # textgrid.write(str(file.with_suffix(".TextGrid.opt")))

    except Exception as e:
        print(file, e)
