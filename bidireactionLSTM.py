import torch
from torch import nn
from deeplearning import dp_torch as dp

# 加载数据
batch_size, num_steps, device = 32, 35, dp.try_gpu()
train_iter, vocab = dp.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = dp.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
dp.train_ch8(model, train_iter, vocab, lr, num_epochs, device)