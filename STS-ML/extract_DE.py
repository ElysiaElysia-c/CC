import scipy.io
import os
import numpy as np

# 提取HBUED数据集的DE微分熵特征，并保存为.mat文件，供后续训练使用

extrac_path = 'HBUED/Dataset/feature/'  # 存放原始特征的路径
save_path = 'HBUED/Dataset/DE/'         # 存放提取后DE特征的路径

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"路径不存在，已创建：{save_path}")
else:
    print(f"路径已存在：{save_path}")

dir_list = os.listdir(extrac_path) 

# label = scipy.io.loadmat(label_path)
# label = label['label'][0]

for f in dir_list:
    if 'mat' not in f:                  # 只处理.mat文件
        continue

    S = scipy.io.loadmat(extrac_path + f)
    DE = S['data'].transpose(0, 2, 1)           # ---------交换第2维和第3维
    label_v = S['valence_labels']   # 效价
    label_a = S['arousal_labels']   # 唤醒度

    mdic = {"DE": DE, "label_v": label_v, "label_a": label_a}

    scipy.io.savemat(save_path + f, mdic)
    print(extrac_path + f, '->', save_path + f)
