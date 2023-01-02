import os
import numpy as np
import tqdm
from sklearn.cluster import KMeans
from utils.tools import get_cn_hubert_units, load_cn_model

zh_hubert_model = load_cn_model()


def get_cn_hubert(path):
    c = get_cn_hubert_units(zh_hubert_model,path=str(path)).squeeze(0).transpose(0,1).cpu().numpy()
    return c

data_base = "/Volumes/Extend/下载/hubert-main/genshindata"
hubert_base = "/Volumes/Extend/下载/hubert-main/units/"
# 遍历dataset目录下所有wav文件
wav_paths = []
for root, dirs, files in os.walk(data_base):
    for file in files:
        if file.endswith('.wav'):
            wav_paths.append(os.path.join(root, file))

# 获取所有wav文件的hubert特征

# wav_paths = [i.replace(".npy", "") for i in os.listdir(hubert_base)]


hubert_units = []
for wav_path in tqdm.tqdm(wav_paths):
    hubert_unit_path = hubert_base + wav_path.split('/')[-1] + '.npy'
    # 先判断是否已经计算过该wav文件的hubert特征
    if os.path.exists(hubert_unit_path):
        # 如果已经计算过，则直接读取缓存结果
        hubert_unit = np.load(hubert_unit_path)
    else:
        # 如果没有计算过，则调用utils.tools.get_cn_hubert_units计算
        hubert_unit = get_cn_hubert(wav_path)
        # 将计算结果缓存到hubert目录下
        np.save(hubert_unit_path, hubert_unit)
    hubert_units.append(hubert_unit)
hubert_units = np.concatenate(hubert_units, axis=0)
