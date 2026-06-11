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
    """r2 score regression loss function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    ss_res = np.sum((y_true - y_predicted) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


    