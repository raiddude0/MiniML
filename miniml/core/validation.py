import numpy as np


def as_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X

def as_1d(y):
    y = np.asarray(y)
    if y.ndim != 1:
        return y.ravel()
    return y

def check_X_y(X, y):
    X = as_2d(X)
    y = as_1d(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} elements")

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be numeric")

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must be numeric")

    if np.isnan(X).any():
        raise ValueError("X contains NaN values")

    if np.isnan(y).any():
        raise ValueError("y contains NaN values")

    return X, y

def check_learning_rate(learning_rate):
    if not isinstance(learning_rate, (float, int)):
        raise ValueError("learning_rate must be a float or int")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return float(learning_rate)
    
def check_epochs(epochs):
    if not isinstance(epochs, int):
        raise ValueError("epochs must be an integer")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    return epochs
