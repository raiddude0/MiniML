import numpy as np
from miniml.metrics.classification import accuracy

def cross_val_score(model, X, y, k=5, metric=None, seed=None):
    """
    Perform k-fold cross validation.
        Parameters
    ----------
    model : object

    X : array-like of shape (n_samples, n_features)

    y : array-like of shape (n_samples,)
    k : int, default=5
        Number of folds.
    
    metric : callable, default=None

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    scores : list of float
        Scores for each fold.
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
#set up seed
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
#create folds
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1 #if n_samples%k != 0, we add 1 to first fold

    scores = []
    current = 0
#loop over folds and calc scores
    for fold_size in fold_sizes:
        #set up test fold
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        #train folds
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        current = stop

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #calc score
        if metric is None:
            score = accuracy(y_test, y_pred)  # default: accuracy
        else:
            score = metric(y_test, y_pred)

        scores.append(score)

    return np.array(scores)
