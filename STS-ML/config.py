import torch

# 配置信息
epochs = 20
batch_size = 4 # 4  ,32  仓库里面是32，论文中是4
lr_list = [[1, 1e-3, 1]]
weight_decay = 8e-5   #1e-4 , 8e-5  
drop_rate = 0.25
num_workers = 0
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
K = 2


# 20个训练轮次，批量大小为4，初始学习率为 1 x 10 - 3，权重衰减为8 x 10 - 5，丢弃率为0.25