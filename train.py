# -*- coding: utf-8 -*-
# @Time     : 2024/10/14 16:03
# @Author   : wangDx
# @File     : main.py
# @describe :基于LeNet-5的手写体识别之训练数据

from torchvision import datasets, transforms
from model import LeNet
import os
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader

# 1.是否使用GPU进行训练数据
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2.设置超参数
EPOCHS = 16  # 训练轮次
BATCH_SIZE = 256  # 一轮训练批量大小

# 构建pipline 对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 数据标准化 提高模型泛化能力 加速模型训练
])
# 3.下载训练数据集
train_dataset = datasets.MNIST(root="./data/train", train=True, transform=pipeline, download=True)

# 4.配置训练数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle 将训练集顺序打乱

# 5.将模型加载到设备上
model = LeNet().to(DEVICE)

# 6.定义损失函数和优化器
loss_fn = CrossEntropyLoss()  # 交叉熵函数损失
option = optim.Adam(model.parameters())  # Adam 梯度下降法

# 7.训练模型
for epoch in range(EPOCHS):
    '''model.train()是PyTorch 中的方法，用于设置模型的训练模式
    在训练模式下，模型会启用 dropout 和 batch normalization 等正则化方法
    并且可以计算梯度以进行参数更新，同时还可以追踪梯度计算的图。'''
    model.train()
    total = 0
    correct = 0.0
    '''
    在PyTorch中，enumerate 函数并不是PyTorch特有的，而是Python内置的一个函数
    用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。
    这在处理PyTorch中的张量（Tensor）或数据集（Dataset）时特别有用，尤其是在需要迭代数据并跟踪每个数据项的位置时。
    '''
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        option.zero_grad()  # 梯度置零

        # 前向传播
        outputs = model(images.float())
        loss = loss_fn(outputs, labels.long())

        # 反向传播,更新优化器
        option.zero_grad()  # 梯度置零

        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()

        loss.backward()  # loss反向传播计算梯度
        option.step()  # 更新网络参数

        if idx % 1000 == 0:
            print(
                "Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(), 100 * (correct / total)))

# 当没有models文件夹时,要创建文件夹
if not os.path.isdir("models"):
    os.mkdir("models")
# 将训练好的模型保存到models文件夹下，方便后面测试时直接调用
torch.save(model, 'models/model.pth')

print("Model finished training")
