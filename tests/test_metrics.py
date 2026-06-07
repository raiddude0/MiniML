import numpy as np
import pytest
from miniml.metrics.regression import MSE, MAE, RMSE, R2_score
from miniml.metrics.classification import accuracy

def test_regression_metrics():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert MSE(y_true, y_pred) == 0.3333333333333333
    assert MAE(y_true, y_pred) == 0.3333333333333333
    assert RMSE(y_true, y_pred) == 0.5773502691896257
    assert R2_score(y_true, y_pred) == 0.5


def test_r2_constant_target():
    y_true = np.array([3, 3, 3])
    y_pred = np.array([2, 2, 2])
    assert R2_score(y_true, y_pred) == 0.0  # should not crash

def test_accuracy():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.75
