import torch
import torch.nn as nn

# 多个损失函数加权求和，权重参数在训练过程中自动更新
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss  有几个任务损失
        x: multi-task loss           传入多个损失值
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    # 初始化了可学习的权重参数，有几个任务就有几个参数，一开始全是 1，后面训练慢慢自己变。
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        print(self.params)

    # 多任务不确定性加权
    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        for i, loss in enumerate(x):
            if i == length:
                loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            else:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
