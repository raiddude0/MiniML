import numpy as np
import pytest
from miniml.model_selection.split import train_test_split

def test_correct_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    assert X_test.shape[0] == 10  # 20% of 50
    assert X_train.shape[0] == 40
    assert y_test.shape[0] == 10
    assert y_train.shape[0] == 40

def test_reproducibility_with_seed():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    split1 = train_test_split(X, y, test_size=0.3, seed=123)
    split2 = train_test_split(X, y, test_size=0.3, seed=123)

    #identical 
    for a, b in zip(split1, split2):
        np.testing.assert_array_equal(a, b)

def test_no_overlap_between_train_and_test():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, seed=0)

    train_idx = set(y_train.tolist())
    test_idx = set(y_test.tolist())

    assert train_idx.isdisjoint(test_idx)
