import numpy as np
import pytest
from miniml.preprocessing.polynomial_features import PolynomialFeatures

def test_bias_only():
    X = np.array([[2], [3]])
    poly = PolynomialFeatures(degree=1, include_bias=True)
    X_poly = poly.fit_transform(X)
    #original + bias
    assert X_poly.shape == (2, 2)
    assert np.allclose(X_poly[:, 0], X[:, 0])
    assert np.allclose(X_poly[:, 1], 1)

def test_square_features():
    X = np.array([[2, 3],
                  [4, 5]])
    poly = PolynomialFeatures(degree=2, include_bias=False, include_interaction=False)
    X_poly = poly.fit_transform(X)
    #original + squares
    assert X_poly.shape == (2, 4)
    assert np.allclose(X_poly[0], [2, 3, 4, 9])
    assert np.allclose(X_poly[1], [4, 5, 16, 25])

def test_interaction_terms():
    X = np.array([[1, 2, 3]])
    poly = PolynomialFeatures(degree=2, include_bias=False, include_interaction=True)
    X_poly = poly.fit_transform(X)
    # original + squares + interactions
    expected = np.array([[1, 2, 3, 1, 4, 9, 2, 3, 6]])
    assert np.allclose(X_poly, expected)

def test_invalid_degree():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=0).fit_transform(np.array([[1,2]]))
