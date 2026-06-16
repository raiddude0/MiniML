import pytest
import numpy as np
from miniml.neighbors.knn_classifier import KNNClassifier
from miniml.neighbors.knn_regressor import KNNRegressor

#--------------------------------------------------tests for KNNClassifier-----------------------------------------------

#prediction on simple data
def test_correct_prediction_simple():
    X_train = [[0,0], [1,1], [2,2]]
    y_train = ['A', 'A', 'B']
    knn = KNNClassifier(n_neighbors=3).fit(X_train, y_train)
    assert knn.predict([1.5, 1.5]) == 'A'

#Tie handling
def test_tie_handling_weighted():
    X_train = [[0,0], [2,2]]
    y_train = ['A', 'B']
    knn = KNNClassifier(n_neighbors=2).fit(X_train, y_train)
    
    assert knn.predict([0.9, 0.9]) == 'A'

#Invalid n_neighbors 
def test_invalid_n_neighbors():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)

# Prediction before fit 
def test_predict_before_fit():
    knn = KNNClassifier(n_neighbors=3)
    with pytest.raises(ValueError):
        knn.predict([1,1])

#multi-feature input
def test_multi_feature_input():
    X_train = [[0,0,0], [1,1,1], [2,2,2]]
    y_train = ['A', 'B', 'B']
    knn = KNNClassifier(n_neighbors=2).fit(X_train, y_train)
    assert knn.predict([1.1, 1.1, 1.1]) == 'B'


#--------------------------------------------------tests for KNNRegressor-----------------------------------------------

def test_correct_prediction_simple2():
    X_train = [[0], [2], [4]]
    y_train = [10, 20, 40]
    knn = KNNRegressor(n_neighbors=2).fit(X_train, y_train)
    pred = knn.predict([1])
    assert pytest.approx(pred, rel=1e-2) == 15

def test_weighted_average():
    X_train = [[0], [10]]
    y_train = [0, 100]
    knn = KNNRegressor(n_neighbors=2).fit(X_train, y_train)
    pred = knn.predict([1])
    assert pred < 50

def test_invalid_n_neighbors2():
    with pytest.raises(ValueError):
        KNNRegressor(n_neighbors=0)

def test_predict_before_fit2():
    knn = KNNRegressor(n_neighbors=3)
    with pytest.raises(ValueError):
        knn.predict([1])

def test_multi_feature_input2():
    X_train = [[0,0], [1,1], [2,2]]
    y_train = [0, 10, 20]
    knn = KNNRegressor(n_neighbors=2).fit(X_train, y_train)
    pred = knn.predict([1.1, 1.1])
    assert 10 <= pred <= 20
