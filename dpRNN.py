import torch
from torch import nn
from deeplearning import dp_torch as dp

batch_size, num_steps = 32, 35
train_iter, vocab = dp.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = torch.device('cpu')
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = dp.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
dp.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)