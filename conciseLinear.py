#生成数据
#读取数据
#初始化参数
#定义模型，损失函数， 优化

import numpy as np
from  torch.utils import data
import torch
from deeplearning import  torch as dp
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = dp.synthetic_data(true_w, true_b, 1000)
batch_size = 10;

data_iter = dp.load_array((features, labels), batch_size, True)
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l=loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')