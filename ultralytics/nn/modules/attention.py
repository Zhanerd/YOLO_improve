# SeNet通道注意力机制

import torch
from torch import nn


class SeNet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeNet, self).__init__()  # 初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        # 两次全连接
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 全局平均池化
        # b,c,h,w -> b,c,1,1
        # b,c,1,1 -> b,c // ratio
        avg = self.avg_pool(x).view([b, c])
        # 两次全连接
        # b,c // ratio -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])  # 获得权值

        return x * fc  # 权值乘原特征图
