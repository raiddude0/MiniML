import numpy as np
import pytest
from miniml.linear_model.LinearRegression import LinearRegression
from miniml.linear_model.RidgeRegression import ridge_regression

def make_dataset():
    # y = 5*x1 + 0.1*x2 + noise
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 2))
    y = 5 * X[:, 0] + 0.1 * X[:, 1] + rng.normal(scale=0.5, size=200)
    return X, y

def test_ridge_shrinks_coefficients():
    X, y = make_dataset()

    # Linear regression 
    linreg = LinearRegression(learning_rate=0.05, epochs=500)
    linreg.fit(X, y)

    # Ridge regression alpha(lambda) > 0
    ridgereg = ridge_regression(learning_rate=0.05, epochs=500, alpha=10.0)
    ridgereg.fit(X, y)

    # compare coef norms
    lin_norm = np.linalg.norm(linreg.m)
    ridge_norm = np.linalg.norm(ridgereg.m)

    assert ridge_norm < lin_norm, (
        f"Expected ridge regression coefficients to be smaller, "
        f"got ridge_norm={ridge_norm:.4f}, lin_norm={lin_norm:.4f}"
    )
