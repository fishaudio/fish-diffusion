<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Fish Diffusion

<div>
<a href="https://github.com/fishaudio/fish-diffusion/actions/workflows/ci.yml">
<img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/fishaudio/fish-diffusion/ci.yml?style=flat-square&logo=GitHub"/>
</a>
<a href="https://hub.docker.com/r/lengyue233/fish-diffusion">
<img alt="Docker Hub" src="https://img.shields.io/docker/cloud/build/lengyue233/fish-diffusion?style=flat-square&logo=Docker&logoColor=white"/>
</a>
<a href="https://huggingface.co/spaces/fishaudio/fish-diffusion">
<img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Spaces-HiFiSinger-blue.svg?style=flat-square"/>
</a>
</div>

<div>
<a href="https://discord.gg/wbYSRBrW2E">
<img alt="Discord" src="https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a href="https://space.bilibili.com/23195420">
<img alt="BiliBili" src="https://img.shields.io/badge/BiliBili-%E5%86%B7%E6%9C%882333-00A1D6?logo=bilibili&style=flat-square&logoColor=white"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/fishaudio/fish-diffusion/blob/main/notebooks/train.ipynb">
<img alt="Open In Colab" src="https://img.shields.io/static/v1?label=Colab&message=Notebook&color=F9AB00&logo=googlecolab&style=flat-square"/>
</a>
</div>

</div>

------

一个简单易懂的 TTS / SVS / SVC 框架.

> 从阅读 [Wiki](https://fishaudio.github.io/fish-diffusion/) 开始! 

## Fish Diffusion 使用条款
1. 获取授权和知识产权侵权：用户应自行负责获取其培训过程中所使用的任何数据集的必要授权，并对因使用输入源而产生的任何侵权问题承担全部责任。Fish Diffusion 及其开发者不承担因使用未授权数据集而可能产生的任何问题的责任。

2. 适当归属：任何基于 Fish Diffusion 的衍生作品都必须明确表示项目及其许可证。在分发 Fish Diffusion 的代码或发布由该项目生成的结果时，用户必须注明原始作者和源代码（Fish Diffusion）。

3. 音频视觉内容及 AI 生成披露：使用 Fish Diffusion 创建的所有衍生作品，包括音频或视频素材，必须明确表示使用了 Fish Diffusion 项目，并声明内容是 AI 生成的。如果使用第三方发布的视频或音频，必须提供原始链接。

4. 同意条款：通过继续使用 Fish Diffusion，用户明确同意本文档中所述的条款和条件。Fish Diffusion 及其开发者不对可能出现的任何后续问题承担责任。


## 简介
基于 DiffSinger 实现歌声音色转换。相较于原 diffsvc 仓库，本仓库优缺点如下
+ 支持多说话人
+ 本仓库代码结构更简单易懂, 模块全部解耦
+ 声码器同样使用 [44.1khz Diff Singer 社区声码器](https://openvpi.github.io/vocoders/)
+ 支持多机多卡训练, 支持半精度训练, 拯救你的训练速度和显存

## 环境准备
以下命令需要在 python 3.10 的 conda 环境下执行

```bash
# 安装 PyTorch 相关核心依赖, 如果已安装则跳过
# 参考 https://pytorch.org/get-started/locally/
conda install "pytorch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 PDM 依赖管理工具, 如果已安装则跳过
# 参考 https://pdm.fming.dev/latest/
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

# 安装依赖 (推荐)
pdm sync

# 如果 PDM 不可用, 或者速度较慢, 可以使用 pip 安装依赖
pip install -r requirements.txt
pip install -e .
```

## 声码器准备
Fish Diffusion 需要 [FishAudio NSF-HiFiGAN](https://github.com/fishaudio/fish-diffusion/releases/tag/v2.0.0) 声码器来生成音频.

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
下载 [Fish Diffusion Release](https://github.com/fishaudio/fish-diffusion/releases/tag/v2.0.0) 中的 `nsf_hifigan-stable-v1.zip`  
解压 `nsf_hifigan` 文件夹到 `checkpoints` 目录下 (如果没有则创建)

如果你想手动下载 [ContentVec](https://github.com/auspicious3000/contentvec), 你可以从 [这里](https://github.com/fishaudio/fish-diffusion/releases/download/v1.12/content-vec-best-legacy-500.pt) 下载, 并将其放入 `checkpoints` 目录下.
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
python tools/preprocessing/extract_features.py --config configs/svc_hubert_soft.py --path dataset/train --clean
```

## 基本训练
> 该项目仍在积极开发, 请记得备份你的 config 文件  
> 该项目仍在积极开发, 请记得备份你的 config 文件  
> 该项目仍在积极开发, 请记得备份你的 config 文件

```bash
# 单机单卡 / 单机多卡训练
python tools/diffusion/train.py --config configs/svc_hubert_soft.py

# 多机训练
python tools/diffusion/train.py --config configs/svc_content_vec_multi_node.py
# 需要在每个节点上配置环境变量，具体请见 https://pytorch-lightning.readthedocs.io/en/1.6.5/clouds/cluster.html

# 继续训练
python tools/diffusion/train.py --config configs/svc_hubert_soft.py --resume [checkpoint file]

# 微调预训练模型
# 注意: 你应该调整配置文件中的学习率调度器为 warmup_cosine_finetune
python tools/diffusion/train.py --config configs/svc_cn_hubert_soft_finetune.py --pretrained [checkpoint file]
```

## 推理
```bash
# 命令行推理, 你可以使用 --help 查看更多参数
python tools/diffusion/inference.py --config [config] \
    --checkpoint [checkpoint file] \
    --input [input audio] \
    --output [output audio]


# Gradio Web 推理, 其他参数会被转为 Gradio 默认参数
python tools/diffusion/inference.py --config [config] \
    --checkpoint [checkpoint file] \
    --gradio
```

## 将 DiffSVC 模型转换为 Fish Diffusion 模型
```bash
python tools/diffusion/diff_svc_converter.py --config configs/svc_hubert_soft_diff_svc.py \
    --input-path [DiffSVC ckpt] \
    --output-path [Fish Diffusion ckpt]
```

## 参与本项目
如果你有任何问题, 请提交 issue 或 pull request.  
你应该在提交 pull request 之前运行 `pdm run lint`

实时预览文档
```bash
pdm run docs
```


## 参考项目
+ [diff-svc original](https://github.com/prophesier/diff-svc)
+ [diff-svc optimized](https://github.com/innnky/diff-svc/)
+ [DiffSinger](https://github.com/openvpi/DiffSinger/) [Paper](https://arxiv.org/abs/2105.02446)
+ [so-vits-svc](https://github.com/innnky/so-vits-svc)
+ [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch) [Paper](https://arxiv.org/pdf/2203.02395.pdf)
+ [CookieTTS](https://github.com/CookiePPP/cookietts/tree/master/CookieTTS/_4_mtw/hifigan)
+ [HiFi-GAN](https://github.com/jik876/hifi-gan) [Paper](https://arxiv.org/abs/2010.05646)
+ [Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## 感谢所有贡献者作出的努力

<a href="https://github.com/fishaudio/fish-diffusion/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/fish-diffusion" />
</a>
