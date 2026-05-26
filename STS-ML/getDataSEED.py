import os
import numpy as np
from scipy import io as scio
from sklearn.model_selection import LeaveOneGroupOut

path = r'./data/SEED/data_independent/DE/time/session1'
label_path = r'./data/SEED/label.mat'
label = scio.loadmat(label_path)['label'].flatten()
print(label)
print("已得到标签")

label = np.array([
    1, 0, -1, -1, 0,
    1, -1, 0, 1, 1,
    0, -1, 0, 1, -1
])

label_map = {-1: 0, 0: 1, 1: 2}
label = np.array([label_map[x] for x in label])

def create_sequence(data, labels, seq_len=10):
    seq_data = []
    seq_label = []

    for i in range(len(data) - seq_len + 1):
        # data: (time, 62, 5)
        x = data[i:i+seq_len]          # (seq_len, 62, 5)
        x = np.transpose(x, (1, 0, 2)) # (62, seq_len, 5)

        y = labels[i + seq_len - 1]
        seq_data.append(x)
        seq_label.append(y)

    return np.array(seq_data), np.array(seq_label)

def data_mev2(seq_len=10):
    all_data = []
    all_label = []
    all_subject = []
    for subject in range(1, 16):
        filename = f'train_dataset_{subject}.mat'
        file_path = os.path.join(path, filename)

        print(f'Loading Subject {subject}')
        mat_data = scio.loadmat(file_path)

        # 这里改为：trial 内分别做滑窗，再拼接 trial 的序列
        subject_seq_data = []
        subject_seq_label = []

        for trial in range(1, 16):
            key = f'de_LDS{trial}'
            data = mat_data[key]                 # (62, time, 5)
            data = np.transpose(data, (1, 0, 2))  # (time, 62, 5)
            num_samples = data.shape[0]
            trial_label = np.full((num_samples,), label[trial - 1])

            # ===== trial 内单独做 sequence（避免跨 trial 混窗）=====
            trial_seq_data, trial_seq_label = create_sequence(
                data, trial_label, seq_len=seq_len
            )

            # 有些 trial 可能 time < seq_len，会变成空，这里跳过以免后面 concatenate 报错
            if len(trial_seq_label) == 0:
                continue

            subject_seq_data.append(trial_seq_data)
            subject_seq_label.append(trial_seq_label)

        # 拼接该被试的所有 trial 的序列样本
        subject_data = np.concatenate(subject_seq_data, axis=0)
        subject_label = np.concatenate(subject_seq_label, axis=0)

        print('data shape:', subject_data.shape)   # (Nsubj, 62, seq_len, 5)
        print('label shape:', subject_label.shape) # (Nsubj,)

        all_data.append(subject_data)
        all_label.append(subject_label)

        # 记录每个序列样本的被试ID
        all_subject.extend([subject] * len(subject_label))

    # 拼接所有被试
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_subject = np.array(all_subject)

    print('\nFinal Shape')
    print(all_data.shape)     # (N, 62, seq_len, 5)
    print(all_label.shape)    # (N,)
    print(all_subject.shape)  # (N,)
    return all_data, all_label, all_subject


def data_SEED(people):
    # Leave-One-Subject-Out
    all_data, all_label, all_subject = data_mev2()
    logo = LeaveOneGroupOut()

    people = people - 1  # 因为people是从1开始的，而logo.split得到的fold是从0开始的，所以要-1
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(all_data, all_label, groups=all_subject)
    ):
        # 仍然按 fold 手动跑
        # 根据fold的值选择对应的训练和测试数据 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
        if fold == people:
            train_data = all_data[train_idx]
            train_label = all_label[train_idx]
            test_data = all_data[test_idx]
            test_label = all_label[test_idx]

            print(f'\nFold {fold + 1}')
            print('Train:', train_data.shape)
            print('Test :', test_data.shape)
            break

    return train_data, train_label, test_data, test_label