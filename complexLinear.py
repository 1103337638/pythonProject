from deeplearning import dp_torch as dp
import torch
import  matplotlib.pyplot as plt
import random

#生成数据集
#读取数据集
#初始化参数
#初始化模型，损失函数，优化
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = dp.synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

w = torch.normal(0, 0.01, size=(2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad=True)

lr =0.03
net = dp.linereg
loss= dp.squared_loss
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        dp.sgd([w, b],lr , batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')



