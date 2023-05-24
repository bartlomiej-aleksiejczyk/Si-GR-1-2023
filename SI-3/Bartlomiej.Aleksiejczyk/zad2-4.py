import numpy as np
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from roughsets_base import roughset_dt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
   return sigmoid(x) * (1 - sigmoid(x))


class Backprop:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.bias1 = np.zeros((1, self.hidden_dim))
        self.weights2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.bias2 = np.zeros((1, self.output_dim))
    def train(self, x, y):
        for i in range(100000):
            # do przodu
            hidden_layer_activation = np.dot(x, self.weights1) + self.bias1
            hidden_layer_output = sigmoid(hidden_layer_activation)
            output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
            y_pred = sigmoid(output_layer_activation)
            # do tyłu
            error = y_pred - y
            d_output = error * sigmoid_derivative(output_layer_activation)
            error_hidden = np.dot(d_output, self.weights2.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_layer_activation)
            self.weights2 -= self.learning_rate * np.dot(hidden_layer_output.T, d_output)
            self.bias2 -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.weights1 -= self.learning_rate * np.dot(x.T, d_hidden)
            self.bias1 -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def predict(self,x):
        hidden_layer_activation = np.dot(x, self.weights1) + self.bias1
        hidden_layer_output = sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
        return sigmoid(output_layer_activation)

# Zbiór treningowy dla XOR
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2 neurony wejścia 4 ukryte i 1 wyjścia
xor = Backprop(2, 4, 1)

# trenuj sieć
xor.train(x, y)

#test sieci
print(xor.predict([0 ,0]))
print(xor.predict([0 ,1]))
print(xor.predict([1 ,0]))
print(xor.predict([1 ,1]))