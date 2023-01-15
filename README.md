# Fish Diffusion
基于[DiffSinger非官方仓库](https://github.com/keonlee9420/DiffSinger) 实现的 [diffsvc](https://github.com/prophesier/diff-svc) 的优化版本 [diff-svc](https://github.com/innnky/diff-svc/) 的优化版本

> 在这个版本上, 科研狗更好做实验了

## 简介
基于 DiffSinger 实现歌声音色转换。相较于原 diffsvc 仓库，本仓库优缺点如下
+ 支持多说话人
+ 本仓库代码结构更简单易懂, 模块全部解耦
+ 声码器同样使用 [441khz Diff Singer 社区声码器](https://openvpi.github.io/vocoders/)
+ 支持多机多卡训练, 支持半精度训练, 拯救你的训练速度和显存

## 数据集准备
仅需要以以下文件结构将数据集放入 dataset 目录即可

```shell
dataset
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   ├───Lxx-0xx8.wav
│   └───abcd (支持子目录)
|       └───xxx1-xxx1.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
