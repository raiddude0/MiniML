import numpy as np


def sigmoid(z):
        return 1 / (1 + np.exp(-z))


def log_loss(X, y, m, b):
    """loss function for logistic regression"""
    epsilon = 1e-15
    z = X @ m + b
    g = sigmoid(z)
    loss = -np.mean(y * np.log(g + epsilon) + (1 - y) * np.log(1 - g + epsilon))
    return loss

def log_loss_gradient(X, y, m, b):
    """gradient of the log loss function"""
    
    z = X @ m + b
    g = sigmoid(z)
    error = g - y
    m_grad = (1 / len(y)) * X.T @ error
    b_grad = (1 / len(y)) * np.sum(error)
    return m_grad, b_grad
    

def accuracy(y_true, y_predicted):
    """accuracy classification score function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    return float(np.mean(y_true == y_predicted))



