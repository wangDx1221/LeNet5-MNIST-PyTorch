# -*- coding: utf-8 -*-
# @Time     : 2024/10/15 17:48
# @Author   : wangDx
# @File     : example.py
# @describe : 基于LeNet-5的手写体识别之举例

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1.是否使用GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2.设置超参数
EPOCHS = 10  # 训练轮次
BATCH_SIZE = 256  # 一轮训练批量大小

# 3.下载并配置数据集
test_dataset = datasets.MNIST(root="F:/pycharm/Project/LeNet5-MNIST-PyTorch/data", train=False,
                              transform=transforms.ToTensor(), download=True)

# 4.配置数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5.加载模型
model = torch.load('models/model.pth')

def show_predict():
    # 预测结果图像可视化
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    plt.figure(figsize=(8, 8))

    for i in range(9):
        (images, labels) = next(iter(loader))
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        title = f"Predicted: {predicted[0]}, True: {labels[0]}"
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].squeeze().to("cuda:0").cpu(), cmap="gray")
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()

show_predict()