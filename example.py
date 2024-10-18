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
BATCH_SIZE = 256  # 一轮训练批量大小

# 3.下载并配置数据集
test_dataset = datasets.MNIST(root="F:/pycharm/Project/LeNet5-MNIST-PyTorch/data", train=False,
                              transform=transforms.ToTensor(), download=True)

# 4.配置数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5.加载模型Al
model = torch.load('models/model.pth')


def show_predict():
    # 预测结果图像可视化
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    '''
    Matplotlib中的 pyplot.figure() 函数的作用就是创建一个图像
        num:图像编号或名称，数字为编号 ，字符串为名称。不指定调用figure时就会默认从1开始。
        figsize:指定figure的宽和高，单位为英寸
        dpi参数指定绘图对象的分辨率，即每英寸多少个像素
        facecolor:背景颜色
        edgecolor:边框颜色
        frameon:是否显示边框
    '''
    plt.figure(figsize=(8, 8))
    for i in range(9):
        (images, labels) = next(iter(loader))

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        title = f"Predicted: {predicted[0]}, True: {labels[0]}"

        '''
        在 Matplotlib 中 plt.subplot() 就是用来创建单个子图的
            nrows 表示 subplot 的行数
            ncols 表示 subplot 的列数
            sharex 表示 subplot 中 x 轴的刻度，所有的 subplot x 轴应该保持相同的刻度
            sharey 表示 subplot 中 y 轴的刻度，所有的 subplot y 轴应该保持相同的刻度
        '''
        plt.subplot(3, 3, i + 1)
        '''
        plt.imshow()函数来显示图像
        cmap='gray' 参数将图像转换为灰度模式
        cmap='hot' 参数将图像转换为热力图
        '''
        plt.imshow(images[0].squeeze().to("cuda:0").cpu(), cmap="gray")
        plt.title(title)
        '''
        在 Matplotlib 中 plt.ticks() 函数 表示的是刻度
        plt.xticks() 就表示x 轴刻度，plt.yticks() 就表示y 轴刻度。
        '''
        plt.xticks([])
        plt.yticks([])

    plt.show()


show_predict()
