import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        '''
        :param gamma: 优化那些难学的样本，控制降权程度(对简单样本降权),gamma越大对简单样本的打压越大，当gamma为0时，退化为交叉熵损失
        :param alpha: alpha控制重点优化那些频率较低的label，alpha的设置跟标签个数相关，一般为数组，对应每个标签的权重
        :param size_average: 控制是否平均
        '''
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # (N, C, H, W) => (N, C, H*W)
            input = input.view(input.size(0), input.size(1), -1)
            # (N, C, H*W) => (N, H*W, C)
            input = input.transpose(1, 2)
            # (N, H*W, C) => (N*H*W, C)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        # 计算pt
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            # 根据label，选择对应的label权重: gather函数，第一个参数为dim，第二个参数为输入行向量index，来替换alpha的行索引
            at = self.alpha.gather(0, target.data.view(-1))
            # log_pt * alpha
            logpt = logpt * Variable(at)
        # focal loss 计算公式
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            # 控制是否平均
            return loss.mean()
        else:
            return loss.sum()