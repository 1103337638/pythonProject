import torch
from deeplearning import dp_torch as dp

batch_size = 256
train_iter, test_iter = dp.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)#这里不应该是小批量嘛，这里是-1的话会自动匹配匹配到合适的维度，所以这里是小批量

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

lr = 0.1

def updater(batch_size):
    return dp.sgd([W, b], lr, batch_size)

num_epochs = 10
dp.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

dp.predict_ch3(net, test_iter)