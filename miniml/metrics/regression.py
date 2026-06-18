import numpy as np

def MSE(y_true, y_predicted):
    """mean squared error regression loss function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    return float(np.mean((y_true - y_predicted) ** 2))



def MSE_gradient(X, y, y_pred):
    error = y_pred - y
    m_grad = (2 / len(y)) * X.T @ error      
    b_grad = (2 / len(y)) * np.sum(error)    
    return m_grad, b_grad




def MAE(y_true, y_predicted):
    """mean absolute error regression loss function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    return float(np.mean(np.abs(y_true - y_predicted)))

def RMSE(y_true, y_predicted):
    """ root mean squarred error regression loss function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_predicted.shape != y_true.shape :
        raise ValueError("y_true and y_predicted must have the same shape")
    
    return float(np.sqrt(np.mean((y_true - y_predicted) ** 2)))

def R2_score(y_true, y_predicted):
    """r2 score regression metric function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    ss_res = np.sum((y_true - y_predicted) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0
    
    return float(1 - ss_res / ss_tot)

def adjusted_R2_score(y_true, y_predicted, n_features):
    """adjusted r2 score regression metric function"""
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    n_samples = len(y_true)

    r2 = R2_score(y_true, y_predicted)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return float(adjusted_r2)



def lasso_loss(y_true, y_predicted, m, alpha):
    """Lasso regression loss function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    m = np.asarray(m)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    mse_loss = MSE(y_true, y_predicted)
    l1_penalty = alpha * np.sum(np.abs(m))

    return float(mse_loss + l1_penalty)

def lasso_gradient(y_true, y_predicted, X, m, alpha):
    """Lasso regression gradient function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    X = np.asarray(X)
    m = np.asarray(m)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    error = y_predicted - y_true
    m_grad = (2 / len(y_true)) * X.T @ error + alpha * np.sign(m)  
    b_grad = (2 / len(y_true)) * np.sum(error)    
    return m_grad, b_grad


def elastic_net_loss(y_true, y_predicted, m, alpha, l1_ratio):
    """elastic net loss function
        if l1_ratio = 1 => lasso regression
        if l1_ratio = 0 => ridge regression
    """
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    m = np.asarray(m)
    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    mse_loss = MSE(y_true, y_predicted)
    l1_penalty = alpha * l1_ratio * np.sum(np.abs(m))
    l2_penalty = alpha * (1 - l1_ratio) * np.sum(m ** 2)
    penalty = l1_penalty + l2_penalty
    return float(mse_loss + penalty)

def elastic_net_gradient(y_true, y_predicted, X, m, alpha, l1_ratio):
    """elastic net gradient function"""
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    X = np.asarray(X)
    m = np.asarray(m)
    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    error = y_predicted - y_true
    MSE_grad = MSE_gradient(X, y_true, y_predicted)
    l1_gradient = alpha * l1_ratio * np.sign(m)
    l2_gradient = 2 * alpha * (1 - l1_ratio) * m
    m_grad = MSE_grad[0] + l1_gradient + l2_gradient
    b_grad = MSE_grad[1]
    return m_grad, b_grad

