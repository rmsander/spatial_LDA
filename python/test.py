import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
X, Y = mnist.train.next_batch(1)
print(X)
print(Y)
print(X.shape)
print(Y.shape)

