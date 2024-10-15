# -*- coding: utf-8 -*-
# @Time     : 2024/10/15 17:25
# @Author   : wangDx
# @File     : test.py
# @describe :基于LeNet-5的手写体识别之测试数据

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

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

model.eval()  # 切换到评估模式而非训练模式即固定参数
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 1表示输出所在行的最大值，为0时表示输出所在列的最大值
        total += labels.size(0)  # 数据总量
        correct += (predicted == labels).sum().item()  # 总准确个数
    print(f'Accuracy:{(100 * correct / total)}%')


