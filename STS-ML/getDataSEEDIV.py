import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import LeaveOneGroupOut

path = r'./data/SEED_IV/data_independent/DE/time/session1'

# ======================================
# SEED-IV 标签（官方24个trial）
# 四分类:
# 0: neutral
# 1: sad
# 2: fear
# 3: happy

trial_label = np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])

# ======================================
# 构造时间序列
# (N,62,5) -> (N',62,T,5)
# ======================================
def create_sequence(data, label, seq_len=10):

    seq_data = []
    seq_label = []

    for i in range(len(data) - seq_len + 1):

        # (seq_len,62,5)
        x = data[i:i + seq_len]

        # -> (62,seq_len,5)
        x = np.transpose(x, (1, 0, 2))

        seq_data.append(x)

        # 当前trial固定标签
        seq_label.append(label)

    return np.array(seq_data), np.array(seq_label)


# ======================================
# 读取所有被试 
# ======================================
def data_IV():
    all_data = []
    all_label = []
    all_subject = []

    seq_len = 10 # 你可以调整这个值来改变序列长度

    for subject in range(1, 16):
        filename = f'train_dataset_{subject}.mat'
        file_path = os.path.join(path, filename)

        print(f'\nLoading Subject {subject}')
        mat_data = sio.loadmat(file_path)

        subject_seq_data = []
        subject_seq_label = []

        for trial in range(1, 25): # 官方是24个trial，
            key = f'de_LDS{trial}'
            data = mat_data[key]
            data = np.transpose(data, (1, 0, 2))
            label = trial_label[trial - 1]
            trial_seq_data, trial_seq_label = create_sequence(
                data,
                label,
                seq_len=seq_len
            )

            if len(trial_seq_label) == 0:
                continue

            subject_seq_data.append(trial_seq_data)
            subject_seq_label.append(trial_seq_label)    

        # ==================================
        # 拼接24个trial
        # ==================================
        subject_data = np.concatenate(
            subject_seq_data,
            axis=0
        )

        subject_label = np.concatenate(
            subject_seq_label,
            axis=0
        )

        print('data shape:', subject_data.shape)   # (Nsubj, 62, seq_len, 5)
        print('label shape:', subject_label.shape)


        all_data.append(subject_data)
        all_label.append(subject_label)

        all_subject.extend(
            [subject] * len(subject_label)
        )

    # 拼接所有被试
    all_data = np.concatenate(
        all_data,
        axis=0
    )

    all_label = np.concatenate(
        all_label,
        axis=0
    )

    all_subject = np.array(all_subject)

    print('\nFinal Shape')
    print(all_data.shape)
    print(all_label.shape)
    print(all_subject.shape)
    return all_data, all_label, all_subject



# LOSO 跨被试 的函数,
# 根据 fold 的值选择对应的训练和测试数据 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
def data_SEEDIV(people):
    all_data, all_label, all_subject = data_IV()
    logo = LeaveOneGroupOut()
    people = people - 1 # 因为 fold 是从0开始的，所以要减1
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(all_data,all_label,groups=all_subject)
    ):
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
        # train(...)