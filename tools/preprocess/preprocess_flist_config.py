import argparse
import json
import os
from pathlib import Path
from random import shuffle

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_list",
        type=str,
        default="./filelists/train.txt",
        help="path to train list",
    )
    parser.add_argument(
        "--val_list", type=str, default="./filelists/val.txt", help="path to val list"
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="./filelists/test.txt",
        help="path to test list",
    )
    parser.add_argument(
        "--source_dir", type=str, default="./dataset", help="path to source dir"
    )
    args = parser.parse_args()

    train = []
    val = []
    test = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(
        [
            ii
            for ii in os.listdir(args.source_dir)
            if os.path.isdir(os.path.join(args.source_dir, ii))
        ]
    ):
        speaker_path = Path(args.source_dir) / speaker

        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = list(speaker_path.glob("**/*.wav"))
        wavs = [
            str(i)
            for i in wavs
            if i.suffix == ".wav" and not i.name.endswith((".22k.wav", ".16k.wav"))
        ]

        shuffle(wavs)
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]

    shuffle(train)
    shuffle(val)
    shuffle(test)

    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            wavpath = fname
            f.write(wavpath + "\n")

    print("Writing dataset/speakers.json")
    with open("dataset/speakers.json", "w") as f:
        json.dump(spk_dict, f, indent=2)
