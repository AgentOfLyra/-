#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# 自用CNN代码，目前正在实现自定义数量的隐藏层和神经元数量的功能

# In[2]:


# 如果采用sigmoid函数（g(z)）的导数是g(z) * (1 - g(z))
def d_sigmoid(x):
    return x * (1 - x)


# In[3]:


class CNN(object):
    def __init__(self, learning_rate, input_cells, output_cells,train_dataset, target,  test_dataset, test_target):
        self.layers_output = []
        self.learning_rate = learning_rate
        self.input_cells = input_cells
        self.output_cells = output_cells
        # 训练集，不包含目标值
        self.train_dataset = train_dataset
        # 训练集目标值
        self.target = target
        # 测试集
        self.test_dataset = test_dataset
        # 测试集目标值
        self.test_target = test_target
        self.theta = []

    # 创建神经网络，初始化权重矩阵
    def create_net(self, hidden_layers):
        self.layers = hidden_layers + 2
        # hidden_layers 表示隐藏层的数量， 而hidden_cells_(number)存储了对应每个隐藏层的数量，而theta表示从第上一层到下一层的权重，初值使用正态分布生成
        # 以下代码可优化 (尴尬，原本是用exec生成一堆变量的，感觉和用列表没啥区别……感觉用列表的话这部分代码还可以用向量乘来简化)
        self.cells = [self.input_cells]
        for i in range(hidden_layers):
            # i表示第几个xx层
            self.cells.append(int(input("input" + str(hidden_layers) + "numbers")))
        self.cells.append(self.output_cells)
        for i in range(1, self.layers):
            self.theta.append(np.random.normal(size=(self.cells[i], self.cells[i - 1])))
            # self.theta.append(np.ones(shape=(self.cells[i], self.cells[i - 1])) * 0.5)
            # 0, np.power(self.cells[i], -0.5),
            # 以上代码可优化

    # 正向传播过程
    def output_session(self, input_values):
        #每层的输入都是是x * 1的，然后每层到下一层的变量的size是(下层的神经元数量) * (这层的神经元数量)
        self.layers_output = [np.array(input_values, ndmin=2).T]
        # 第二层开始到最后一层（也就是第一个隐藏层的输入值，存在一个问题，万一没有输入隐藏层？
        for i in range(0, self.layers - 1):
            layers_input = np.dot(self.theta[i], self.layers_output[i])
            self.layers_output.append(expit(layers_input))
    # 反向传播算法
    # 梯度下降
    def back_propagation(self, target_values):
        err = [target_values - self.layers_output[self.layers - 1]]
        # print(err)
        for i in range(1, self.layers - 1):
            # print(err[i])
            err.append(np.dot(self.theta[-i].T, err[i - 1]))
        for i in range(self.layers - 1):
            self.theta[self.layers - i - 2] += self.learning_rate * np.dot(err[i] * d_sigmoid(self.layers_output[self.layers - i - 1]), self.layers_output[self.layers - i - 2].T)
            # print(self.theta[-i - 1])


    def normalization(self, case=1):
        # 标准归一化
        if case == 0:
            self.train_dataset = (self.train_dataset - self.train_dataset.min()) / (self.train_dataset.max() - self.train_dataset.min())
            self.test_dataset = (self.test_dataset - self.test_dataset.min()) / (self.test_dataset.max() - self.test_dataset.min())
        # 神经网络归一化，要求全大于等于1
        elif case == 1:
            self.train_dataset = np.log10(self.train_dataset + 1) / np.log10(self.train_dataset.max() + 1)
            self.test_dataset = np.log10(self.test_dataset + 1) / np.log10(self.test_dataset.max() + 1)
        elif case == 2:
            self.train_dataset = self.train_dataset / 255 * 0.99 + 0.01
            self.test_dataset = self.test_dataset / 255 * 0.99 + 0.01
        else:
            print(" ")
    # 训练神经网络
    def train(self):
        lines = self.train_dataset.shape[0]
        for i in range(lines):
            self.output_session(self.train_dataset[i])
            self.back_propagation(self.target[i])
    # 使用测试集进行测试
    def test(self):
        lines = self.test_dataset.shape[0]
        results = []
        cnt = 0
        for i in range(lines):
            self.output_session(self.test_dataset[i])
            results.append(np.argmax(self.layers_output[-1]))
            if results[i] == np.argmax(self.test_target[i]):
                cnt += 1
        accuracy = cnt / lines * 100
        return results, accuracy

