import  torch
from torch import nn
from deeplearning import dp_torch as dp

# 读取数据


# 初始化参数 #定义模型  #定义激活函数
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)
net.apply(init_weights)






#初始化超参数
batch_size, lr, num_epochs = 256, 0.1, 10
#定义损失函数

loss = nn.CrossEntropyLoss(reduction='none')
#定义优化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = dp.load_data_fashion_mnist(batch_size)




#训练
dp.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


