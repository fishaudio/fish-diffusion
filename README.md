<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Fish Diffusion

<div>
<a href="https://github.com/fishaudio/fish-diffusion/actions/workflows/ci.yml">
<img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/fishaudio/fish-diffusion/ci.yml?style=flat-square&logo=GitHub">
</a>
<a href="https://hub.docker.com/r/lengyue233/fish-diffusion">
<img alt="Docker Hub" src="https://img.shields.io/docker/cloud/build/lengyue233/fish-diffusion?style=flat-square&logo=Docker&logoColor=white">
</a>
<a href="https://colab.research.google.com/drive/1GPNq1FWH5LE2f79M4QV2UbdWWazfgrpt">
<img alt="Colab" src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=Google%20Colab&style=flat-square&logoColor=white">
</a>
</div>

<div>
<a href="https://discord.gg/wbYSRBrW2E">
<img alt="Discord" src="https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square">
</a>
<a href="https://space.bilibili.com/23195420">
<img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-%E5%86%B7%E6%9C%882333-00A1D6?logo=bilibili&style=flat-square&logoColor=white">
</a>
<img alt="QQ" src="https://img.shields.io/badge/QQ-588056461-EB1923?logo=Tencent%20QQ&style=flat-square">
</div>

</div>

------

一个简单易懂的 TTS / SVS / SVC 框架.

> 从阅读 [Wiki](https://fishaudio.github.io/fish-diffusion/) 开始!

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

# 安装依赖 (推荐)
poetry install

# 如果 Poetry 不可用, 或者速度较慢, 可以使用 pip 安装依赖
pip install -r requirements.txt
pip install -e .
```

## 声码器准备
Fish Diffusion 需要 [OPENVPI 441khz NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1) 声码器来生成音频.

### 自动下载
```bash
python tools/download_nsf_hifigan.py
```

如果你的训练环境处于国内, 可以使用 [GitHub Proxy](https://ghproxy.com/) 来加速下载.

```bash
python tools/download_nsf_hifigan.py --use-ghproxy
```

如果你正在使用脚本自动化训练, 可以使用传参 `--agree-license` 的方式同意 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证.

```bash
python tools/download_nsf_hifigan.py --agree-license
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
# 提取全部数据的特征, 如 pitch, text features, mel features 等
python tools/preprocessing/extract_features.py --config configs/svc_hubert_soft.py --path dataset --clean
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

# 微调预训练模型
# 注意: 你应该调整配置文件中的学习率调度器为 warmup_cosine_finetune
python train.py --config configs/svc_cn_hubert_soft_finetune.py --pretrained [checkpoint]
```

## 推理
```bash
# 命令行推理, 你可以使用 --help 查看更多参数
python inference.py --config [config] \
    --checkpoint [checkpoint] \
    --input [input audio] \
    --output [output audio]


# Gradio Web 推理, 其他参数会被转为 Gradio 默认参数
python inference.py --config [config] \
    --checkpoint [checkpoint] \
    --gradio
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

实时预览文档
```bash
sphinx-autobuild docs docs/_build/html
```


## 参考项目
+ [diff-svc original](https://github.com/prophesier/diff-svc)
+ [diff-svc optimized](https://github.com/innnky/diff-svc/)
+ [DiffSinger](https://github.com/openvpi/DiffSinger/)
+ [SpeechSplit](https://github.com/auspicious3000/SpeechSplit)

## 感谢所有贡献者作出的努力

<a href="https://github.com/fishaudio/fish-diffusion/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/fish-diffusion" />
</a>
