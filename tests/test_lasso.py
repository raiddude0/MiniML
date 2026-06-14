import numpy as np
import pytest
from miniml.linear_model import lasso_regression

LassoRegression = getattr(lasso_regression, "LassoRegression", None)
pytestmark = pytest.mark.skipif(
    LassoRegression is None,
    reason="LassoRegression implementation is missing",
)

def test_lasso_shrinks_coefficients():
    # Dataset with a strong linear relationship
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    # Low regularization
    model_small_alpha = LassoRegression(learning_rate=0.1, epochs=500, alpha=0.01)
    model_small_alpha.fit(X, y)

    # High regularization
    model_large_alpha = LassoRegression(learning_rate=0.1, epochs=500, alpha=10.0)
    model_large_alpha.fit(X, y)

    # Coefficient should be smaller with stronger regularization
    assert abs(model_large_alpha.m[0]) < abs(model_small_alpha.m[0])

def test_lasso_can_zero_unimportant_feature():
    # Two features, only the first one matters
    X = np.array([[1, 0],
                  [2, 0],
                  [3, 0],
                  [4, 0]])
    y = np.array([3, 5, 7, 9])

    model = LassoRegression(learning_rate=0.1, epochs=1000, alpha=1.0)
    model.fit(X, y)

    # The second coefficient should be close to zero
    assert abs(model.m[1]) < 1e-3

def test_lasso_loss_decreases():
    # Simple dataset
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LassoRegression(learning_rate=0.1, epochs=200, alpha=0.01)
    model.fit(X, y)

    # Loss should decrease over epochs
    assert model.loss_history[0] > model.loss_history[-1]

def test_lasso_predict_shape():
    # Dataset with matching labels
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    model = LassoRegression(learning_rate=0.1, epochs=200, alpha=0.01)
    model.fit(X, y)

    preds = model.predict(X)
    # Predictions should have the same shape as y
    assert preds.shape == y.shape

def test_predict_before_fit_raises():
    # predict() should raise an error if fit() has not been called
    model = LassoRegression()
    X = np.array([[1], [2]])
    with pytest.raises(ValueError):
        model.predict(X)
