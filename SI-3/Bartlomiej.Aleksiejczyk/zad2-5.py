import numpy as np
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Backprop:
    #tak jak było w poleceniu learning rate jest domyśleni ustawiony na 0.1
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights1 = np.array([
            [0.1,0,0.3 ],
            [-0.2,0.2,-0.4]
        ])
        self.bias1 = np.array([[0.1, 0.2, 0.5]])
        self.weights2 = np.array([
            [-0.4,0.2],
            [0.1,-0.1],
            [0.6,-0.2],
        ])
        self.bias2 = np.array([[-0.1, 0.2]])

    def train(self, x, y):
        for i in range(2000):
            # Propagacja w przód
            hidden_layer_activation = np.dot(x, self.weights1) + self.bias1
            hidden_layer_output = sigmoid(hidden_layer_activation)
            output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
            y_pred = sigmoid(output_layer_activation)
            # Popagacja w tył
            error = y_pred - y
            d_output = error * sigmoid_derivative(output_layer_activation)
            error_hidden = np.dot(d_output, self.weights2.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_layer_activation)
            # Modyfikacja wrtości
            self.weights2 -= self.learning_rate * np.dot(hidden_layer_output.T, d_output)
            self.bias2 -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.weights1 -= self.learning_rate * np.dot(x.T, d_hidden)
            self.bias1 -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def predict(self,x):
        # Propagacja w przód
        hidden_layer_activation = np.dot(x, self.weights1) + self.bias1
        hidden_layer_output = sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
        return sigmoid(output_layer_activation)

# Zbiór treningowy
x = np.array([[0.6, 0.1], [0.2, 0.3]])
y = np.array([[1], [0]])
#Tworzenie sieci neuronowej z 2 neuronami wejścia, 3 schowanymi i 2 wyjścia.
xor = Backprop(2, 3, 2)

print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))

print("Propagacja w tył")
print("Nietknięte wagi i biasy: ")
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
xor.train(x, y)
print("Postęp po 2000 epochów")
print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))
print(xor.predict([0.2, 0.3]))
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
xor.train(x, y)
print("Postęp po 4000 epochów")
print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))
print(xor.predict([0.2, 0.3]))
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
xor.train(x, y)
print("Postęp po 6000 epochów")
print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))
print(xor.predict([0.2, 0.3]))
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
xor.train(x, y)
print("Postęp po 8000 epochów")
print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))
print(xor.predict([0.2, 0.3]))
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
xor.train(x, y)
print("Postęp po 10000 epochów")
print("Wynik propagacji w przód:")
print(xor.predict([0.6, 0.1]))
print(xor.predict([0.2, 0.3]))
print("Weights 1")
print(xor.weights1)
print("Weights 2")
print(xor.weights2)
print("Bias 1")
print(xor.bias1)
print("BIas 2")
print(xor.bias2)
