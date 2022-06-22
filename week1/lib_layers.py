import autograd.numpy as np

from utils import *


class Layer:
    def apply(self, x):
        return x


class Function(Layer):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.func = get_func(name, params)

    def apply(self, x):
        return self.func(x)


class Linear(Layer):
    def __init__(self, weights, bias, name):
        self.weights = weights.transpose(1, 0)
        self.bias = bias.reshape(-1, bias.size)
        self.func = get_func(name, None)

    def apply(self, x):
        if self.func == None:
            return x @ self.weights + self.bias
        else:
            return self.func(x @ self.weights + self.bias)
