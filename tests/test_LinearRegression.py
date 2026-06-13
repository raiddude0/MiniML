import numpy as np
import pytest
from miniml.linear_model.LinearRegression import LinearRegression

def test_linear_regression_learns_simple_relation():
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = 2 * X.flatten() + 3

    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Check parameters
    assert np.allclose(model.m, [2.0], atol=0.1)
    assert np.allclose(model.b, 3.0, atol=0.1)

    # Check predictions
    preds = model.predict(np.array([[6], [7]]))
    assert np.allclose(preds, [15.0, 17.0], atol=0.1)

def test_loss_decreases_over_epochs():
    X = np.array([[1], [2], [3], [4], [5]])
    y = 2 * X.flatten() + 3

    model = LinearRegression(learning_rate=0.01, epochs=200)
    model.fit(X, y)

    # check loss history
    assert len(model.loss_history) == 200
    assert model.loss_history[0] > model.loss_history[-1]

def test_predict_shape_matches_input():
    X = np.array([[1], [2], [3]])
    y = 2 * X.flatten() + 3

    model = LinearRegression(learning_rate=0.01, epochs=500)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (3,)
