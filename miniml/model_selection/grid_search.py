import numpy as np
from .cross_validation import cross_val_score

def grid_search(model, params, X, y, metric, k=5, seed=None, greater_is_better=True):
    if not params:
        raise ValueError("Parameter grid is empty.")

    best_score = -np.inf if greater_is_better else np.inf
    best_params = None

    for p in params:
        model.set_params(**p)
        scores = cross_val_score(model, X, y, k=k, metric=metric, seed=seed)
        mean_score = np.mean(scores)

        if greater_is_better:
            if mean_score > best_score:
                best_score, best_params = mean_score, p
        else:
            if mean_score < best_score:
                best_score, best_params = mean_score, p

    return best_params, best_score


