import matplotlib.pyplot as plt
import math
import time
import numpy as np
import torch
from  torch.utils import  data
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize;

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    ## axes 坐标轴
    # fmts就是线条的样式
    # figsize是图的大小
    # xlabel是横坐标名称
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()#这里应该是关于坐标轴的设置，主要是用来操作坐标轴的

    # 如果X有一个轴，输出True,这里应该是判断X 是不是张量，若是且x的维度为1，或者X是个List
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)# 这里就是x的每个元素✖️y的长度
    axes.cla()#用于清除当前坐标轴
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt) #这里直接画了3个图
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start

    def start(self):
        """记录开始"""
        self.tik=time.time()

    def stop(self):
        self.times.append(time.time()-self.tik)
        return  self.time[-1]
    def avg(self):
        return sum(self.times)/len(self.times)

    def sum(self):
        return   sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_example):
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)

def linereg(X, w, b):
    return  torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad / batch_size
                param.grad.zero_()

def load_array(data_arrays, batch_size, is_train):
    """"pytorch的数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)

