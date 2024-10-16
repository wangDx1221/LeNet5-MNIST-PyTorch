# -*- coding: utf-8 -*-
# @Time     : 2024/10/14 16:03
# @Author   : wangDx
# @File     : main.py
# @describe :基于LeNet-5的手写体识别之构建网络模型

from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(  # nn.Sequential:相当于一个容器，将一系列操作包含其中
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # out_size(6*28*28)
            nn.ReLU(),  # ReLu()激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # out_size(6*14*14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # out_size(16*10*10)
            nn.ReLU(),  # ReLu()激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # out_size(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    print("Model Create Finished!")
