import numpy as np
from .cross_validation import cross_val_score

def grid_search(model, params, X, y, metric, k=5, seed=None):
    """ Perform grid search for hyperparameter tuning.
    for simplicity we only use one hyperparameter here"""
    best_score = -np.inf
    best_params = None
    for p in params:
        model.set_params(**p)
        scores = cross_val_score(model, X, y, k=5, metric=metric, seed=seed)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = p

    return best_params, best_score


