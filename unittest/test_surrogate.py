import sys

import numpy as np
import pytest

sys.path.insert(0, "./")

from bayes_optim.surrogate import GaussianProcess, RandomForest, SurrogateAggregation, trend
from bayes_optim.surrogate.gaussian_process.gpr import l1_cross_distances
from sklearn.metrics import r2_score

np.random.seed(12)


def test_pickling():
    n_sample = 110
    levels = ["OK", "A", "B", "C", "D", "E"]
    X = np.c_[np.random.randn(n_sample, 2).astype(object), np.random.choice(levels, size=(n_sample, 1))]
    y = np.sum(X[:, 0:-1] ** 2.0, axis=1) + 5 * (X[:, -1] == "OK")

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    # sklearn-random forest
    rf = RandomForest(levels={2: levels}, max_features="sqrt")
    rf.fit(X_train, y_train)
    y_hat, mse = rf.predict(X_test, eval_MSE=True)

    print("sklearn random forest:")
    print("target :", y_test)
    print("predicted:", y_hat)
    print("MSE:", mse)
    print("r2:", r2_score(y_test, y_hat))
    print()

    X = np.c_[
        np.random.randn(n_sample, 2).astype(object),
        np.random.choice(levels, size=(n_sample, 1)),
    ]
    y = np.sum(X[:, 0:-1] ** 2.0, axis=1) + 5 * (X[:, -1] == "OK")

    X_train, y_train = X[:100, :], y[:100]
    X_test, y_test = X[100:, :], y[100:]

    rf_ = RandomForest(levels={2: levels}, max_features="sqrt")
    rf_.fit(X_train, y_train)

    rf_aggr = SurrogateAggregation((rf, rf_), weights=(0.1, 0.9))
    y_hat, mse = rf_aggr.predict(X_test, eval_MSE=True)

    print("sklearn random forest:")
    print("target :", y_test)
    print("predicted:", y_hat)
    print("MSE:", mse)
    print("r2:", r2_score(y_test, y_hat))


def test_multioutput_gpr():
    n_sample = 100
    dim = 10
    X = np.random.rand(n_sample, dim)
    y = np.sum(X**2.0, axis=1)
    thetaL, thetaU = 1e-10 * np.ones(dim), 10 * np.ones(dim)

    # noiseless mode
    model = GaussianProcess(
        theta0=np.random.rand(dim) * (thetaU - thetaL) + thetaL,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=0,
        noise_estim=False,
        optimizer="BFGS",
        wait_iter=3,
        random_start=dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )
    model.fit(X, y)
    model.predict(X, eval_MSE=True)

    # noisy mode
    model = GaussianProcess(
        theta0=np.random.rand(dim) * (thetaU - thetaL) + thetaL,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=1e-6,
        noise_estim=False,
        optimizer="BFGS",
        wait_iter=3,
        random_start=dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )
    model.fit(X, y)
    model.predict(X, eval_MSE=True)

    # noise estimation mode
    model = GaussianProcess(
        theta0=np.random.rand(dim) * (thetaU - thetaL) + thetaL,
        thetaL=thetaL,
        thetaU=thetaU,
        nugget=1e-6,
        noise_estim=True,
        optimizer="BFGS",
        wait_iter=3,
        random_start=dim,
        likelihood="concentrated",
        eval_budget=100 * dim,
    )
    model.fit(X, y)
    model.predict(X, eval_MSE=True)


@pytest.mark.parametrize("noise_estim, nugget", [(True, 1e-6), (False, 0.05), (False, 1e-6)])
def test_gpr_noisy_prediction(noise_estim, nugget):
    """compare the GPR prediction to the exact posterior (ordinary Kriging) with the same hyperparameters"""
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.15, 40)
    model = GaussianProcess(
        mean=trend.constant_trend(2),
        corr="matern",
        thetaL=1e-3 * np.ones(2),
        thetaU=1e3 * np.ones(2),
        nugget=nugget,
        noise_estim=noise_estim,
        optimizer="BFGS",
        wait_iter=3,
        random_start=20,
        likelihood="concentrated",
        eval_budget=400,
        random_state=1,
    )
    model.fit(X, y)
    X_test = np.vstack([X[:3] + 0.01, rng.random((3, 2))])
    mean, mse = model.predict(X_test, eval_MSE=True)

    # the exact posterior
    sigma2, noise_var = np.ravel(model.sigma2)[0], np.ravel(model.noise_var)[0]
    R0 = model.corr(model.theta_, l1_cross_distances(X, Y=X)).reshape(len(X), len(X))
    K_inv = np.linalg.inv(sigma2 * R0 + noise_var * np.eye(len(X)))
    k = sigma2 * model.corr(model.theta_, l1_cross_distances(X_test, Y=X)).reshape(len(X_test), len(X))
    F = np.ones((len(X), 1))
    FKF = (F.T @ K_inv @ F).item()
    beta = (F.T @ K_inv @ y).item() / FKF
    u = 1 - F.T @ K_inv @ k.T
    mean_ = beta + k @ K_inv @ (y - beta)
    mse_ = sigma2 - np.einsum("ij,jk,ik->i", k, K_inv, k) + u.ravel() ** 2 / FKF

    assert np.allclose(np.ravel(mean), mean_, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.ravel(mse), mse_, rtol=1e-6, atol=1e-10)
    assert np.all(np.ravel(mse) > 0)

    # the gradient against finite differences
    x, delta = np.array([0.37, 0.61]), 1e-6
    y_dx, mse_dx = model.gradient(x)
    for i, e in enumerate(np.eye(2)):
        (m1, s1), (m2, s2) = [model.predict(np.atleast_2d(x + c * delta * e), eval_MSE=True) for c in (1, -1)]
        assert np.isclose(np.ravel(y_dx)[i], (m1 - m2).item() / (2 * delta), rtol=1e-4, atol=1e-6)
        assert np.isclose(np.ravel(mse_dx)[i], (s1 - s2).item() / (2 * delta), rtol=1e-3, atol=1e-7)
