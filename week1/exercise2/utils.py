import autograd.numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def get_func(name, params):
    if name == 'relu':
        return relu
    elif name == 'sigmoid':
        return sigmoid
    elif name == 'tanh':
        return tanh
