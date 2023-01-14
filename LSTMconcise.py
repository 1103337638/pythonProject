import torch
from torch import nn
from deeplearning import dp_torch as dp

batch_size, num_steps = 32, 35
train_iter, vocab = dp.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, dp.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = dp.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
dp.train_ch8(model, train_iter, vocab, lr, num_epochs, device)