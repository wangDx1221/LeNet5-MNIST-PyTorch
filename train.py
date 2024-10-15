# -*- coding: utf-8 -*-
# @Time     : 2024/10/14 16:03
# @Author   : wangDx
# @File     : main.py
# @describe :基于LeNet-5的手写体识别之训练数据

from torchvision import datasets, transforms
from model import LeNet
import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

# 1.是否使用GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2.设置超参数
EPOCHS = 10  # 训练轮次
BATCH_SIZE = 256  # 一轮训练批量大小
LR = 1e-1  # 学习率

# 3.下载并配置数据集
train_dataset = datasets.MNIST(root="F:/pycharm/Project/LeNet5-MNIST-PyTorch/data", train=True,
                               transform=transforms.ToTensor(), download=True)

# 4.配置数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 5.将模型加载到设备上
model = LeNet().to(DEVICE)

# 6.定义损失函数和优化器
loss_fn = CrossEntropyLoss()
sgd = SGD(model.parameters(), lr=LR)

# 7.训练模型
total_step = len(train_loader)  # 总训练次数

for epoch in range(EPOCHS):
    model.train()
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向传播
        outputs = model(images.float())
        loss = loss_fn(outputs, labels.long())

        # 反向传播,更新优化器
        sgd.zero_grad()  # 梯度置零
        loss.backward()  # loss反向传播计算梯度
        sgd.step()  # 更新网络参数

        if (idx + 1) % 100 == 0:
            print(f'Epoch[{epoch + 1}/{EPOCHS}], Step[{idx + 1}/{total_step}], Loss:{loss.item():.4f}')

if not os.path.isdir("models"):
    os.mkdir("models")
torch.save(model, 'models/model.pth')

print("Model finished training")
