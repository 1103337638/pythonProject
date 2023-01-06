import torch
from torch import  nn
from deeplearning import dp_torch as dp
#读取数据
batch_size = 256
train_iter, test_iter = dp.load_data_fashion_mnist(batch_size=batch_size)
#定义模型
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

#初始化参数
#Sequential应该有内置的初始参数，若要自己设置初始化就需要用net.apply
#看一下内置的初始化参数
print(net[0].weight.data[0])
#定义损失函数

#定义优化器

#训练
lr, num_epochs = 0.9, 10
dp.train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('mps'))
#正确率