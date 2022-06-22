import autograd.numpy as np

class Model:
    def __init__(self, shape, lower, upper, layers):
        self.shape = shape
        self.lower = lower
        self.upper = upper
        self.layers = layers


    def apply(self, x):
        output = x

        for layer in self.layers:
            output = layer.apply(output)

        return output
