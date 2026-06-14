import numpy as np
import pytest
from miniml.linear_model.logistic_regression import logistic_regression as LogisticRegression
from miniml.metrics.classification import accuracy

def test_logistic_regression_accuracy_improves():
    #linearly separable dataset
    X = np.array([[-2], [-1], [0], [1], [2]])
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, epochs=200)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy(y, preds)

    # Accuracy should be high for this dataset
    assert acc >= 0.8

    # Loss should decrease 
    assert model.loss_history[0] > model.loss_history[-1]

def test_predict_proba_range():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, epochs=100)
    model.fit(X, y)

    proba = model.predict_proba(X)

    # probabetween 0 and 1
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
