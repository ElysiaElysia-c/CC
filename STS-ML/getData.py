import os
import numpy as np
from scipy import io as scio
from sklearn.model_selection import LeaveOneGroupOut

path = r'./data/SEED/data_independent/DE/time/session1'
label_path = r'./data/SEED/label.mat'
label = scio.loadmat(label_path)['label'].flatten()
print(label)
print("已得到标签")    # label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
# =========================
# 标签（SEED官方）
# 15个视频对应标签
# 0: negative
# 1: neutral
# 2: positive
# =========================
label = np.array([
    1, 0, -1, -1, 0,
    1, -1, 0, 1, 1,
    0, -1, 0, 1, -1
])
# 转成 0,1,2
label_map = {
    -1: 0,
    0: 1,
    1: 2
}
label = np.array([label_map[x] for x in label])

def create_sequence(data, labels, seq_len=10):

    seq_data = []
    seq_label = []
    for i in range(len(data) - seq_len + 1):
        # (62,10,5)
        x = data[i:i+seq_len]
        # 原本:
        # (10,62,5)
        # 转成:
        # (62,10,5)
        x = np.transpose(x, (1,0,2))
        y = labels[i + seq_len - 1]
        seq_data.append(x)
        seq_label.append(y)
    return np.array(seq_data), np.array(seq_label)


def data_me():
    all_data = []
    all_label = []
    all_subject = []
    for subject in range(1, 16):
        filename = f'train_dataset_{subject}.mat' # 每个被试文件的名字  train_dataset_1.mat
        file_path = os.path.join(path, filename) # 完整路径    

        print(f'Loading Subject {subject}')

        mat_data = scio.loadmat(file_path)

        subject_data = []
        subject_label = []

        # 15个trial
        for trial in range(1, 16):

            key = f'de_LDS{trial}'

            # shape:
            # (62, time, 5)
            data = mat_data[key]

            # 转成:
            # (time, 62, 5)
            data = np.transpose(data, (1, 0, 2))

            # 当前trial长度
            num_samples = data.shape[0]

            # 当前trial标签
            trial_label = np.full(
                (num_samples,),
                label[trial - 1]
            )

            subject_data.append(data)
            subject_label.append(trial_label)


        # 拼接15个trial
        subject_data = np.concatenate(subject_data, axis=0)
        subject_label = np.concatenate(subject_label, axis=0)
        subject_data, subject_label = create_sequence(
                subject_data,
                subject_label,
                seq_len=10
            )

        print('data shape:', subject_data.shape)
        print('label shape:', subject_label.shape)

        all_data.append(subject_data)
        all_label.append(subject_label)

        # 记录被试ID
        all_subject.extend(
            [subject] * len(subject_label)
        )

    # =========================
    # 拼接所有被试
    # =========================
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_subject = np.array(all_subject)

    print('\nFinal Shape')
    print(all_data.shape)
    print(all_label.shape)
    print(all_subject.shape)

    # 最终:
    # all_data:
    # (N, 62, 5)

    # all_label:
    # (N,)

    # =========================
    # Leave-One-Subject-Out
    # 跨被试划分
    # =========================
    logo = LeaveOneGroupOut()
    # =========================
    # 读取所有被试
    # =========================

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(all_data, all_label, groups=all_subject)):
        # 根据fold的值选择对应的训练和测试数据 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
        if fold == 0:
            train_data = all_data[train_idx]
            train_label = all_label[train_idx]

            test_data = all_data[test_idx]
            test_label = all_label[test_idx]

            print(f'\nFold {fold + 1}')
            print('Train:', train_data.shape)
            print('Test :', test_data.shape)
            break

    return train_data, train_label, test_data, test_label
        # 这里开始训练
        # train(...)  
    