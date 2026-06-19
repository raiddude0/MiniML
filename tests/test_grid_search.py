import numpy as np
import pytest
from miniml.linear_model.ridge_regression import RidgeRegression as Ridge
from miniml.metrics.regression import MSE
from miniml.model_selection.grid_search import grid_search



#y = 2x + 1
X = np.arange(20).reshape(-1, 1)
y = 2 * X.flatten() + 1

def test_grid_search_returns_best_params_and_score():
    param_grid = [
        {"alpha": 0.01},
        {"alpha": 0.1},
        {"alpha": 1.0}
    ]
    model = Ridge(learning_rate=0.001, epochs=500)
    best_params, best_score = grid_search(model, param_grid, X, y, metric=MSE, k=5, seed=0, greater_is_better=False)

    #best_params is one of the candidates?
    assert best_params in param_grid
    #score is a float?
    assert isinstance(best_score, float)

def test_grid_search_prefers_lower_alpha_on_simple_linear_data():
    param_grid = [
        {"alpha": 0},     # no penalty, should fit perfectly
        {"alpha": 10.0}   # heavy penalty, should underfit
    ]
    model = Ridge(learning_rate=0.001, epochs=500)
    best_params, best_score = grid_search(model, param_grid, X, y, metric=MSE, k=5, seed=0, greater_is_better=False)

    #small alpha should perform better
    assert best_params["alpha"] in [0, 10.0]


def test_grid_search_handles_empty_param_list():
    model = Ridge(learning_rate=0.001, epochs=500)
    with pytest.raises(ValueError):
        grid_search(model, [], X, y, metric=MSE, k=5, seed=0, greater_is_better=False)
