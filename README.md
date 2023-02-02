# Fish Diffusion
![CI Status](https://github.com/fishaudio/fish-diffusion/actions/workflows/ci.yml/badge.svg)
[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&logo=discord&logoColor=white)](https://discord.gg/3CtFhTsvuG)

一个简单易懂的 TTS / SVS / SVC 框架.

[English Document](README.en.md)

## 简介
基于 DiffSinger 实现歌声音色转换。相较于原 diffsvc 仓库，本仓库优缺点如下
+ 支持多说话人
+ 本仓库代码结构更简单易懂, 模块全部解耦
+ 声码器同样使用 [441khz Diff Singer 社区声码器](https://openvpi.github.io/vocoders/)
+ 支持多机多卡训练, 支持半精度训练, 拯救你的训练速度和显存

## 环境准备
以下命令需要在 python 3.10 的 conda 环境下执行

```bash
# 安装 PyTorch 相关核心依赖, 如果已安装则跳过
# 参考 https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装 Poetry 依赖管理工具, 如果已安装则跳过
# 参考 https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 安装依赖
poetry install
```

## 声码器准备
Fish Diffusion 需要 [OPENVPI 441khz NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) 声码器来生成音频.

### 自动下载
```bash
python tools/download_nsf_hifigan.py
```

### 手动下载
下载 [441khz 声码器](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) 中的 `nsf_hifigan_20221211.zip`

解压 `nsf_hifigan` 文件夹到 `checkpoints` 目录下 (如果没有则创建)

## 数据集准备
仅需要以以下文件结构将数据集放入 dataset 目录即可

```shell
dataset
├───train
│   ├───xxx1-xxx1.wav
│   ├───...
│   ├───Lxx-0xx8.wav
│   └───speaker0 (支持子目录)
│       └───xxx1-xxx1.wav
└───valid
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

```bash
# 1. 提取全部数据的特征, 如 pitch, text features, mel features 等
python tools/preprocessing/extract_features.py --config configs/svc_hubert_soft.py --path dataset --clean

# 2. 生成训练集统计信息
python tools/preprocessing/generate_stats.py --input-dir dataset/train --output-file dataset/stats.json
```

## 基本训练
> 该项目仍在积极开发, 请记得备份你的 config 文件  
> 该项目仍在积极开发, 请记得备份你的 config 文件  
> 该项目仍在积极开发, 请记得备份你的 config 文件

```bash
# 单机单卡 / 单机多卡训练
python train.py --config configs/svc_hubert_soft.py

# 继续训练
python train.py --config configs/svc_hubert_soft.py --resume [checkpoint]
```

## 推理
```bash
python inference.py --config configs/svc_hubert_soft.py \
    --checkpoint [checkpoint] \
    --input [input audio] \
    --output [output audio]
```

## 将 DiffSVC 模型转换为 Fish Diffusion 模型
```bash
python tools/diff_svc_converter.py --config configs/svc_hubert_soft_diff_svc.py \
    --input-path [DiffSVC ckpt] \
    --output-path [Fish Diffusion ckpt]
```

## 参与本项目
如果你有任何问题, 请提交 issue 或 pull request.  
你应该在提交 pull request 之前运行 `tools/lint.sh`


## 参考项目
+ [diff-svc original](https://github.com/prophesier/diff-svc)
+ [diff-svc optimized](https://github.com/innnky/diff-svc/)
+ [DiffSinger](https://github.com/openvpi/DiffSinger/)
