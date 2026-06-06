import numpy as np
import pytest
from miniml.preprocessing.standard_scaler import StandardScaler

def test_fit_transform_mean_std():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # Column mean ~ 0
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    # Column std ~ 1
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

def test_inverse_transform_recovers_original():
    X = np.array([[10, 20], [30, 40], [50, 60]])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_recovered)

def test_zero_variance_feature_raises():
    X = np.array([[1, 1], [1, 1], [1, 1]])
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.fit(X)

def test_fit_transform_1d_array():
    X = np.array([1, 2, 3, 4, 5])
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    assert np.allclose(X_scaled.mean(), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(), 1, atol=1e-7)
