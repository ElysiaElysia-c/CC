import os
import sys
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.optim as optim
from scipy import io as scio
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import zscore
from model import DE_MoE
from AutoWeight import AutomaticWeightedLoss
from tqdm import tqdm
from utils import eegDataset, calc_diff_loss

from getData import data_me
from getDataSEED import data_SEED
import os
import copy
import config
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './'  # 设置环境变量

# -------------SEED数据集的路径
path = r'./data/SEED/data_independent/DE/time/session1'
label_path = r'./data/SEED/label.mat'

# session_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
session_label = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])
label_map = {
    -1: 0,
    0: 1,
    1: 2
}
session_label = np.array([label_map[x] for x in session_label]) # 将标签映射为0,1,2，适配交叉熵损失

batch_size = config.batch_size
epochs = config.epochs
lr_list = config.lr_list
weight_decay = config.weight_decay
drop_rate = config.drop_rate
device = config.device
num_workers = config.num_workers
DATASETS = ['SEED', 'SEED_IV', 'MPED'] # 三个数据集
DATASET = path.strip().split('/')[-5] # 提取数据集的名字 ，倒数第5的目录名
assert DATASET in DATASETS # 判断数据集的名字是否正确 一定要是要求的三个数据集之一
DEPENDENT = path.strip().split('/')[-4] # 
# if DEPENDENT == 'data_independent':
#     DATASET = DATASET+'_'+DEPENDENT


# 数据加载器
def load_dataloader(data_train, data_test, label_train, label_test):
    train_iter = DataLoader(dataset=eegDataset(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=False)

    test_iter = DataLoader(dataset=eegDataset(data_test, label_test),
                            batch_size=int(data_test.shape[0]/10),
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    return train_iter, test_iter

# 标准化
def eeg_transformer(x):
    return (x - x.mean()) / x.std()

# 训练
def train(train_iter, test_iter, model, model_best, awl, criterion, lr_list, epochs, acc_test_best, people):
    # Train

    print('-----began training on 开始训练 -----', device, '...')

    # acc_test_best = 0.0
    n = 0
    for ep in range(epochs): # 20轮
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        #  lr_list = [[1, 1e-3, 1]]  #分阶段训练，但是我们现在只设置了一个阶段，所以这个循环就相当于没什么用，后续如果想分阶段训练了再来设置这个列表就好了
        for lts, lr, l in lr_list:  #只循环一次 

            if acc_test_best == 1:
                # print('acc_test_best = 1')
                break
            
            for lt in range(int(lts)): # 只循环一次
                
                if acc_test_best == 1:
                    # print('acc_test_best = 1')
                    break
                
                model.testmode = False
                model.train()
                n += 1
                # batch_id = 1
                correct, total, total_loss = 0, 0, 0.
                for ind, data in enumerate(train_iter):
                    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    model.testmode = False
                    model.train()

                    x = data[0].float().to(device)  # [256,62,5]([B,C,F])
                    y = data[1].to(device)

                    # x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, out = model(x)
                    out = model(x)
                    _, pred = torch.max(out, dim=1)
                    correct += sum([1 for a, b in zip(pred, y) if a == b])
                    total += len(y)
                    accuracy = correct/total # 计算当前轮次的累计准确率

                    loss = criterion(out, y.long())

                    optimizer.zero_grad()
                    total_loss += loss
                    loss.backward()
                    optimizer.step() # 更新模型的参数

                    # batch_id += 1 每10个batch打印一次当前的训练状态
                    if (ind+1) % 10 == 0:
                        print('people {}, epoch {}, ind {}, loss: {}, accuracy: {}'.format(
                            people, ep+1, ind + 1, total_loss / ind+1, accuracy))
                        acc_test = evaluate(test_iter, model) # 评估当前模型在测试集上的准确率

                        if acc_test >= acc_test_best:
                            n = 0
                            acc_test_best = acc_test
                            model_best = model
                            checkpoint = {
                                'model_state_dict': model_best.state_dict(),
                                'ACC': acc_test_best
                            }
                            # 保存模型
                            torch.save(
                                checkpoint, f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl')
                        
                        if acc_test_best == 1:
                            print('acc_test_best = 1')
                            break
                
                acc_test = evaluate(test_iter, model)

                if acc_test >= acc_test_best:
                    n = 0
                    acc_test_best = acc_test
                    model_best = model
                    checkpoint = {
                        'model_state_dict': model_best.state_dict(),
                        'ACC': acc_test_best
                    }
                    torch.save(
                        checkpoint, f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl')
                    
                if acc_test_best == 1:
                    print('acc_test_best = 1')
                    break
        
        # 每一轮的结束 打印结果
        print('people {} 每轮的总损失 {}: {}'.format(
            people, ep + 1, total_loss))

        print('>>> 这一轮最好的准确率: {}'.format(acc_test_best))

        if acc_test_best == 1:
            print('acc_test_best = 1')
            break

    return acc_test_best


# 测试评估
def evaluate(test_iter, model):
    # Eval
    print(' 每10个batch测试一次 ...')
    model.testmode = True
    model.eval() # 不更新参数，所以很快
    correct, total = 0, 0
    for x, y in test_iter:
        # Add channels = 1
        x = x.float().to(device)

        # Categogrical encoding
        y = y.to(device)

        out = model(x)

        _, pre = torch.max(out, dim=1)

        correct += sum([1 for a, b in zip(pre, y) if a == b])
        total += len(y)
    print('这次测试的准确率: {}'.format(correct / total))
    return correct / total





# 主执行函数 people是被试的编号，1-15
def runs(people):

    train_data, train_label, test_data, test_label = data_SEED(people)

    # 打印数据的形状
    # train_data = np.expand_dims(train_data, axis=1)
    # test_data = np.expand_dims(test_data, axis=1)
    # print(train_data.shape)
    # print(test_data.shape)


    acc_test_best = 0.0
    # 读取最好的精度，如果存在的话
    if not os.path.exists(f'SGCNN/{DATASET}_checkpoint/independent/session1'):
        os.makedirs(
            f'SGCNN/{DATASET}_checkpoint/independent/session1')
    if os.path.exists(f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl'):
        check = torch.load(
            f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl', weights_only=True)
        # acc_test_best = check['ACC']


    # 分类类别数
    HC = 3 
    #SEED数据集的类别数是3，SEED_IV和MPED是4和7，之前的代码里是根据数据集名字来设置的，现在直接设置为3，因为我们用的是EED数据集，如果换数据集了记得改回来哦！

    awl = AutomaticWeightedLoss(3)

    # zscore标准化
    # test_data = zscore(np.array(test_data, dtype=np.float32))
    test_data = zscore(test_data)  

    model_best = DE_MoE(5, 65, 62, linearsize=512, dropout=drop_rate, testmode=False, HC=HC).to(device)
    model = DE_MoE(5, 65, 62, linearsize=512, dropout=drop_rate, testmode=False, HC=HC).to(device)

    # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
    criterion = nn.CrossEntropyLoss().to(device)

    # train_data = zscore(train_data)
    num = int(train_data.shape[0]/14)
    for i in range(14):
        train_data[i*num:(i+1)*num] = zscore(train_data[i*num:(i+1)*num])


    # 训练的数据要求导，必须使用torch.tensor包装 
    train_data = torch.tensor(train_data)
    # train_label = torch.tensor(train_label)
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.tensor(test_data) 
    # test_label = torch.tensor(test_label)
    test_label = torch.from_numpy(test_label).long()

    train_iter, test_iter = load_dataloader(train_data, test_data, train_label, test_label)
    
    # 训练的结果为 返回最好的准确率
    acc_test_best = train(train_iter, test_iter, model, model_best, awl, criterion, lr_list, epochs, acc_test_best, people) #------------开始训练
    acc_all.append(acc_test_best)
    print('----------------这个被试最好的结果 best test Accuracy: {}'.format( acc_test_best))

    # if people == 15:
    #     acc_mean = np.array(acc_all).mean()
    #     print('>>> LOSV test acc: ', acc_all)
    #     print('>>> LOSV test mean acc: ', acc_mean)
    #     print('>>> LOSV test std acc: ', np.std(np.array(acc_all))) # 标准差


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间（秒）
    acc_all = []
    # runs(1) 
    # runs(2) 
    # runs(3)
    # runs(4)
    # runs(5)
    # runs(6)
    # runs(7)
    runs(8) # ok
    # runs(9)
    runs(10) #ok
    runs(11)
    runs(12)
    runs(13)
    runs(14)
    runs(15)
    end_time = time.time()    # 记录结束时间（秒）
    # 计算总耗时
    total_time = end_time - start_time
    print(f"训练总耗时：{total_time:.2f} 秒")
    # 每次训练的时候，要删除生成的checkpoint文件夹里的之前的模型文件，
    # 不然会直接加载之前的模型，导致训练结果不对！
    # 记得先删除checkpoint文件夹里对应被试的模型文件哦！
    print("\n" + "="*50)
    print("这个被试训练完成！")
    print("这个被试的最优准确率列表：", acc_all)
    print("="*50)
