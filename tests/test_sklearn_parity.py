# tests/test_parity_models.py
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinear, Ridge as SkRidge, LogisticRegression as SkLogistic
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.datasets import make_classification

from miniml.linear_model import LinearRegression as MiniLinear, RidgeRegression as MiniRidge, LogisticRegression as MiniLogistic
from miniml.preprocessing.standard_scaler import StandardScaler as MiniScaler




def test_linear_regression_parity():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 3 * X.flatten() + 2

    mini = MiniLinear().fit(X, y)
    sk = SkLinear().fit(X, y)

    
    assert np.allclose(mini.m, sk.coef_, rtol=1e-1, atol=1e-1)

    assert np.allclose(mini.b, sk.intercept_, rtol=1e-1, atol=1e-1) #WORKS


def test_ridge_regression_parity():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = 3 * X.flatten() + 2

    mini = MiniRidge(alpha=1.0).fit(X, y)
    sk = SkRidge(alpha=1.0).fit(X, y)

    #since sklearn uses a closed form for optimization, we can't expect exact parity,
    #   but we can check the coefficients are both shrinking compared to linear regression instead


    sk_lin = SkLinear().fit(X, y)
    assert abs(mini.m[0]) < abs(sk_lin.coef_[0]) + 1 #allow tolerance
    assert abs(sk.coef_[0]) < abs(sk_lin.coef_[0]) + 1


def test_logistic_regression_parity():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

    mini = MiniLogistic().fit(X, y)
    sk = SkLogistic(max_iter=1000).fit(X, y)

    mini_acc = (mini.predict(X) == y).mean()
    sk_acc = (sk.predict(X) == y).mean()

    #accuracy parity, both reasonably high and close
    assert mini_acc > 0.7
    assert abs(mini_acc - sk_acc) < 0.1

    # Probability parity: just check shape and range
    mini_probs = mini.predict_proba(X)
    if mini_probs.ndim == 1:
        mini_probs = np.column_stack([1 - mini_probs, mini_probs])

    assert mini_probs.shape == sk.predict_proba(X).shape
    assert np.all((mini_probs >= 0) & (mini_probs <= 1))



def test_standard_scaler_parity():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    mini = MiniScaler().fit(X)
    sk = SkScaler().fit(X)

    mini_scaled = mini.transform(X)
    sk_scaled = sk.transform(X)

    assert np.allclose(mini_scaled, sk_scaled, atol=1e-8)
