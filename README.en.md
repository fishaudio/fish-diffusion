# Fish Diffusion
![CI Status](https://github.com/fishaudio/fish-diffusion/actions/workflows/ci.yml/badge.svg)

An easy to understand TTS / SVS / SVC training framework.

[中文文档](README.md)

## Summary
Using Diffusion Model to solve different voice generating tasks. Compared with the original diffsvc repository, the advantages and disadvantages of this repository are as follows:
+ Support multi-speaker
+ The code structure of this repository is simpler and easier to understand, and all modules are decoupled
+ Support [441khz Diff Singer community vocoder](https://openvpi.github.io/vocoders/)
+ Support multi-machine multi-devices training, support half-precision training, save your training speed and memory

## Preparing the environment
The following commands need to be executed in the conda environment of python 3.10

```bash
# Install PyTorch related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Install the project dependencies
poetry install
```

## Vocoder preparation
Download and unzip `nsf_hifigan_20221211.zip` from [441khz vocoder](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1)

Copy `model` file to `checkpoints/nsf_hifigan` directory

## Dataset preparation
You only need to put the dataset into the `dataset` directory in the following file structure

```shell
dataset
├───train
│   ├───xxx1-xxx1.wav
│   ├───...
│   ├───Lxx-0xx8.wav
│   └───speaker0 (Subdirectory is also supported)
│       └───xxx1-xxx1.wav
└───valid
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

```bash
# 1. Extract all data features, such as pitch, text features, mel features, etc.
python tools/preprocessing/extract_features.py --config configs/svc_hubert_soft.py --path dataset --clean

# 2. Generate training set statistics
python tools/preprocessing/generate_stats.py --input-dir dataset/train --output-file dataset/stats.json
```

## Baseline training
> The project is under active development, please backup your config file  
> The project is under active development, please backup your config file  
> The project is under active development, please backup your config file  

```bash
# Single machine single card / multi-card training
python train.py --config configs/svc_hubert_soft.py

# Resume training
python train.py --config configs/svc_hubert_soft.py --resume [checkpoint]
```

## Inference
```bash
python inference.py --config configs/svc_hubert_soft.py \
    --checkpoint [checkpoint] \
    --input [input audio] \
    --output [output audio]
```

## Convert a DiffSVC model to Fish Diffusion
```bash
python tools/diff_svc_converter.py --config configs/svc_hubert_soft_diff_svc.py \
    --input-path [DiffSVC ckpt] \
    --output-path [Fish Diffusion ckpt]
```

## Contributing
If you have any questions, please submit an issue or pull request.  
You should run `tools/lint.sh` before submitting a pull request.

## Credits
+ [diff-svc original](https://github.com/prophesier/diff-svc)
+ [diff-svc optimized](https://github.com/innnky/diff-svc/)
+ [DiffSinger](https://github.com/openvpi/DiffSinger/)
+ [diffusers](https://github.com/huggingface/diffusers)
