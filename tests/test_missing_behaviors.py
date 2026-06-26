import numpy as np
import pytest

from miniml.linear_model import LinearRegression, LogisticRegression, RidgeRegression
from miniml.model_selection.split import train_test_split
from miniml.preprocessing.standard_scaler import StandardScaler


def test_linear_predict_before_fit_raises():
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[1], [2]]))


def test_logistic_predict_before_fit_raises():
    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([[0], [1]]))


def test_ridge_loss_history_has_one_entry_per_epoch():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = RidgeRegression(learning_rate=0.01, epochs=50, alpha=0.1)
    model.fit(X, y)

    assert len(model.loss_history) == 50
    assert model.loss_history[0] > model.loss_history[-1]


def test_train_test_split_without_shuffle_keeps_order():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, shuffle=False)

    assert np.array_equal(X_test.flatten(), np.array([1, 2]))
    assert np.array_equal(y_test, np.array([1, 2]))
    assert np.array_equal(X_train.flatten(), np.array([3, 4, 5]))
    assert np.array_equal(y_train, np.array([3, 4, 5]))


def test_standard_scaler_inverse_transform_returns_original_values():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_original = scaler.inverse_transform(X_scaled)

    assert np.allclose(X_original, X)
