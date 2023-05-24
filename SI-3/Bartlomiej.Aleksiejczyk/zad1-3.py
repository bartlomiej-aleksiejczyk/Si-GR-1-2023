import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from celluloid import Camera
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def main():
    # Regresja liniowa z użyciem schodzenia radientowego
    X = np.array([[6.5], [7.0], [7.4], [8.2], [9.0]])
    y = np.array([[2000], [2002], [2005], [2007], [2010]])
    y = y.flatten()
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=40, tol=1e-3, warm_start=True, random_state=2005))
    loss = []
    fig = plt.figure(figsize=(10, 10))
    labelsize_ = 14
    camera = Camera(fig)
    ax0 = fig.add_subplot(1, 1, 1)
    for epoch in range(125):
        ax0.scatter(X, y, color='b', marker='x', s=44)
        reg.fit(X, y)
        loss.append(mean_squared_error(y, reg.predict(X)))
        leg = ax0.plot(X.flatten(), reg.predict(X).flatten(),
                       color='r', label=str(epoch))
        ax0.set_title("Regresja liniowa", fontsize=25)
        ax0.tick_params(axis='both', which='major', labelsize=labelsize_)
        ax0.set_xlabel("Procent bezrobotnych", fontsize=25, labelpad=10)
        ax0.set_ylabel("Rok", fontsize=25, labelpad=10)
        ax0.tick_params(axis='both', which='major', labelsize=labelsize_)
        ax0.legend(leg, [f'epochs: {epoch * 40}'], loc='upper right', fontsize=15)
        ax0.set_ylim([1998, 2012])
        plt.tight_layout()
        camera.snap()
    animation = camera.animate(interval=5, repeat=False, repeat_delay=500)
    animation.save('SimpleLinReg_1.gif', writer='imagemagick')


main()
