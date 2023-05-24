from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron

#ZNAJDŹ PERCEPTRON AND
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

pp = Perceptron()
pp.n_iter_ = 100

pp.fit(X, y)
print(pp.predict([[0, 0]]))
#[0]
print(pp.predict([[0, 1]]))
#[0]
print(pp.predict([[1, 0]]))
#[0]
print(pp.predict([[1, 1]]))
#[1]

#ZNAJDŹ PERCEPTRON NOT
X = [[0], [1]]
y = [1, 0]

ppn = Perceptron()
ppn.n_iter_ = 100

print('Etykiety klas:', np.unique(y))

ppn.fit(X, y)
print(ppn.predict([[1]]))
#[0]
print(ppn.predict([[0]]))
#[1]