# -*- coding: utf-8 -*-
# @Time     : 2024/10/15 17:25
# @Author   : wangDx
# @File     : test.py
# @describe :基于LeNet-5的手写体识别之测试数据

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

# 1.是否使用GPU进行测试数据
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2.设置超参数
BATCH_SIZE = 256  # 一轮训练批量大小

# 构建pipline 对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化 降低模型复杂度
])
# 3.下载测试数据集
test_dataset = datasets.MNIST(root="F:/pycharm/Project/LeNet5-MNIST-PyTorch/data", train=False,
                              transform=pipeline, download=True)

# 4.配置数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5.加载已经训练好的模型
model = torch.load('models/model.pth')

'''
model.eval()的作用是 不启用 Batch Normalization 和 Dropout 切换到评估模式而非训练模式即固定参数
pytorch 会自动把 BN 和 DropOut 固定住，不会取平均，而是用训练好的值
在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定，神经网络每一次生成的结果也是不固定的，生成质量可能好也可能不好
'''
model.eval()
correct = 0
total = 0
'''
torch.no_grad()是PyTorch的一个上下文管理器，用于在不需要计算梯度的场景下禁用梯度计算。
在使用torch.no_grad()上下文管理器的情况下，所有涉及张量操作的函数都将不会计算梯度，从而节省内存和计算资源。
因为此时用来测试，即不需要调整梯度，所以要禁用梯度，与上文model.eval()作用类似
'''
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        '''
        1.torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）
        2.为什么这里选择用这么特殊的下划线？这是因为我们不关心最大值是什么，而关心最大值对应的index是什么,就选择使用下划线_代表不需要用到的变量
        3.数字1其实可以写为dim=1，1表示输出所在行的最大值，为0时表示输出所在列的最大值
        '''
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)  # 数据总量
        correct += (predicted == labels).sum().item()  # 计算预测和实际索引相符的数据个数

    print(f'Accuracy:{(100 * correct / total)}%')  # 输出准确率


