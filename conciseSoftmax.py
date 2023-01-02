import matplotlib.pyplot as plt
import torch
from torch import nn
from deeplearning import dp_torch as dp

batch_size = 256
train_iter, test_iter = dp.load_data_fashion_mnist(batch_size)

#初始化模型参数,Pytorch 中的 model.apply(fn) 会递归地将函数 fn 应用到父模块的每个子模块以及model这个父模块自身。通常用于初始化模型的参数。
#网络
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#注意nn.Flatten()与torch.Flatten()是不同的，torch的拉平默认为dim=0，而nn.Flatten()默认为dim=1
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
#损失函数
loss = nn.CrossEntropyLoss(reduction='none')

#优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
dp.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()

