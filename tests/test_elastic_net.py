import numpy as np
import pytest
from miniml.linear_model.elastic_net import ElasticNet

def test_elastic_net_fit_and_predict():
    
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6]])
    y = np.array([13, 19, 25, 31, 37])  

    model = ElasticNet(learning_rate=0.01, epochs=200, alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)

    # test correct shape
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # 2. Loss should decrease 
    assert model.loss_history[0] > model.loss_history[-1]

def test_elastic_net_regularization_shrinks_coefficients():
    # strong correlation
    X = np.array([[1, 1],
                  [2, 2],
                  [3, 3],
                  [4, 4]])
    y = np.array([2, 4, 6, 8])  

    #  without regularization
    model_no_reg = ElasticNet(alpha=0.0, epochs=200, learning_rate=0.01)
    model_no_reg.fit(X, y)

    #with regularization
    model_reg = ElasticNet(alpha=10.0, l1_ratio=0.5, epochs=200, learning_rate=0.01)
    model_reg.fit(X, y)

    # coefficients with regularization should have smaller magnitude
    assert np.linalg.norm(model_reg.m) < np.linalg.norm(model_no_reg.m)
