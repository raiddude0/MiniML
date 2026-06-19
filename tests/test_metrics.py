import numpy as np
import pytest
from miniml.metrics.regression import MSE, MAE, RMSE, R2_score, adjusted_R2_score
from miniml.metrics.classification import accuracy, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score



def test_regression_metrics():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert MSE(y_true, y_pred) == 0.3333333333333333
    assert MAE(y_true, y_pred) == 0.3333333333333333
    assert RMSE(y_true, y_pred) == 0.5773502691896257
    assert R2_score(y_true, y_pred) == 0.5


def test_r2_constant_target():
    y_true = np.array([3, 3, 3])
    y_pred = np.array([2, 2, 2])
    assert R2_score(y_true, y_pred) == 0.0  # should not crash

def test_accuracy():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == 0.75

from sklearn.metrics import (
    precision_score as skl_precision,
    recall_score as skl_recall,
    f1_score as skl_f1,
    confusion_matrix as skl_confusion,
    roc_auc_score as skl_auc
) #for testing our metrics against sklearn's implementation not using them



def test_precision_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.isclose(precision_score(y_true, y_pred),
                      skl_precision(y_true, y_pred))

def test_recall_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.isclose(recall_score(y_true, y_pred),
                      skl_recall(y_true, y_pred))

def test_f1_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.isclose(f1_score(y_true, y_pred),
                      skl_f1(y_true, y_pred))

def test_confusion_matrix():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.array_equal(confusion_matrix(y_true, y_pred),
                          skl_confusion(y_true, y_pred))

def test_roc_auc_score():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert np.isclose(roc_auc_score(y_true, y_scores),
                      skl_auc(y_true, y_scores))