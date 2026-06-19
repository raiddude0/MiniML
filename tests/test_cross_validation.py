import numpy as np
from miniml.model_selection.cross_validation import cross_val_score

class DummyModel:
    def fit(self, X, y):
        self.majority = np.bincount(y).argmax()
    def predict(self, X):
        return np.full(len(X), self.majority)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def test_cross_val_score_basic():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)
    model = DummyModel()

    scores = cross_val_score(model, X, y, k=5, metric=accuracy, seed=0)

    assert len(scores) == 5
    assert all(0 <= s <= 1 for s in scores)

def test_cross_val_score_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)
    model = DummyModel()

    s1 = cross_val_score(model, X, y, k=5, metric=accuracy, seed=42)
    s2 = cross_val_score(model, X, y, k=5, metric=accuracy, seed=42)

    assert np.allclose(s1, s2)
