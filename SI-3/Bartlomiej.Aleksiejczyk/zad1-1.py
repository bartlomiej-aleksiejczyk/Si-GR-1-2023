import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import celluloid
from celluloid import Camera
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)
def main():
    #Regresja liniowa z użyciem schodzenia radientowego
    y = np.array([[6.5],[7.0],[7.4],[8.2],[9.0]])

    X = np.array([[2000],[2002],[2005],[2007],[2010]])
    y=y.flatten()

    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000900, tol=1e-3))
    reg.fit(X, y)
    # tutaj należy wpisać dane do wyszukania
    pred = ([[2011], [2012], [2013],[2014],[2015],[2016],[2017],[2018],[2019],[2020],[2021]])
    # wyświetla wynik funckji
    print(reg.predict(pred).round(3))
    # Pipeline(steps=[('standardscaler', StandardScaler()),
    #                 ('sgdregressor', SGDRegressor())])


main()
