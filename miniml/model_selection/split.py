import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    X = np.array(X)
    y = np.array(y)

    n_samples = X.shape[0]
    
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be a float oor an int please")
    
    if n_test<=0 or n_test >= n_samples:
        raise ValueError("test_size must be positive and less than n_samples")
    
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)

    if shuffle:
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, X_test, y_train, y_test
