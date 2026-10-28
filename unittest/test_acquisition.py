import sys

import numpy as np
import pytest

sys.path.insert(0, "./")
from bayes_optim.acquisition.acquisition_fun import UCB
from bayes_optim.surrogate import GaussianProcess, trend


class FixedModel:
    """A stub surrogate returning the prediction and MSE from lookup tables"""

    def __init__(self, y_hat, mse):
        self.y_hat, self.mse = np.asarray(y_hat, dtype=float), np.asarray(mse, dtype=float)

    def predict(self, X, eval_MSE=False):
        idx = np.asarray(X, dtype=int)[:, 0]
        return (self.y_hat[idx], self.mse[idx]) if eval_MSE else self.y_hat[idx]


@pytest.mark.parametrize("minimize", [True, False])
def test_UCB_direction(minimize):
    # point 0 and 1 have the same uncertainty, but point 0 has a better prediction
    y_hat = [1.0, 5.0] if minimize else [5.0, 1.0]
    acq = UCB(model=FixedModel(y_hat, [1.0, 1.0]), alpha=0.5, minimize=minimize)
    assert acq([0]) > acq([1])

    # with the same prediction, the more uncertain point is preferred
    acq = UCB(model=FixedModel([2.0, 2.0], [1.0, 4.0]), alpha=0.5, minimize=minimize)
    assert acq([1]) > acq([0])


@pytest.mark.parametrize("minimize", [True, False])
def test_UCB_gradient(minimize):
    np.random.seed(42)
    dim, lb, ub = 2, -5, 5
    X = np.random.rand(20, dim) * (ub - lb) + lb
    y = np.sum(X**2, axis=1)
    model = GaussianProcess(
        mean=trend.constant_trend(dim),
        corr="matern",
        thetaL=1e-3 * (ub - lb) * np.ones(dim),
        thetaU=1e3 * (ub - lb) * np.ones(dim),
        nugget=1e-6,
        noise_estim=False,
        optimizer="BFGS",
        wait_iter=3,
        random_start=2 * dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )
    model.fit(X, y)
    acq = UCB(model=model, alpha=0.5, minimize=minimize)

    x, delta = np.array([1.3, -2.1]), 1e-5
    _, f_dx = acq(x, return_dx=True)
    fd = [(acq(x + delta * e) - acq(x - delta * e)) / (2 * delta) for e in np.eye(dim)]
    assert np.allclose(np.ravel(f_dx), np.ravel(fd), rtol=1e-3, atol=1e-6)
