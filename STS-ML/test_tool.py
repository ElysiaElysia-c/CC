import torch
import numpy
print(torch.__version__);
print(torch.version.cuda)
print(numpy.__version__)
# print(torch.cuda.is_available())


# 查看保存的准确率数据是多少
# DATASET = 'SEED_IV'
DATASET = 'SEED'
for i in range(1, 16):
    check = torch.load(f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{i}.pkl', weights_only=True)
    print(f'被试{i}的准确率: {check["ACC"]:.4f}')
