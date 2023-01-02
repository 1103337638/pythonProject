import torch
from deeplearning import dp_torch as dp
from torch import nn
#读取数据
batch_size = 256
train_iter, test_iter = dp.load_data_fashion_mnist(batch_size)

batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#初始化模型参数，  torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_inputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_inputs, requires_grad=True))

params = [W1, b1, W2, b2]
# 定义激活函数,这里的torch.max如何应用广播机制？？
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)
#定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)#@为矩阵乘法
    return  (H@W2 + b2)

#损失函数
loss = nn.CrossEntropyLoss(reduction='none') #reduction：用来指定损失结果返回的是mean、sum还是none。
# 训练

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
dp.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
dp.predict_ch3(net, test_iter)