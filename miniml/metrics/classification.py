import numpy as np

def accuracy(y_true, y_predicted):
    """accuracy classification score function"""

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_true.shape != y_predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape")
    
    return float(np.mean(y_true == y_predicted))



