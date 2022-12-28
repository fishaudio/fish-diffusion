# diff-svc
基于[DiffSinger非官方仓库](https://github.com/keonlee9420/DiffSinger) 实现的 [diffsvc](https://github.com/prophesier/diff-svc)

>（没错又复活了，之前导致弃坑的bug似乎修复了）\
>（暂时依然在开发测试中）

## 简介
基于Diffsinger + softvc 实现歌声音色转换。相较于原diffsvc仓库，本仓库优缺点如下
+ 支持多说话人
+ 本仓库基于非官方diffsinger仓库修改实现，代码结构更加简单易懂
+ 声码器同样使用 [441khz diffsinger社区声码器](https://openvpi.github.io/vocoders/)
+ 不支持加速

## 数据集准备
仅需要以以下文件结构将数据集放入dataset_raw目录即可
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## 数据预处理
整体基本类似sovits3.0
1. 重采样
```shell
python resample.py
 ```
2. 自动划分训练集 验证集 测试集
```shell
python preprocess_flist_config.py
```
3. 生成hubert、f0、mel与stats
```shell
python preprocess_hubert_f0.py && python gen_stats.py
```

执行完以上步骤后 dataset 目录便是预处理完成的数据，可以删除dataset_raw文件夹，
也可以删除重采样后的临时wav文件`rm dataset/*/*.wav`

## 训练
```shell
python3 train.py --model naive --dataset ms --restore_step RESTORE_STEP 
```

## 推理
[inference.py](inference.py)