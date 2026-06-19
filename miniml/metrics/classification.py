import numpy as np

def sigmoid(z):
        return 1 / (1 + np.exp(-z))


def log_loss(X, y, m, b):
    """Binary cross-entropy loss function for logistic regression (BCE)"""
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


def euclidean_distance(a,b):
    """euclidean distance function for classification"""

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    
    return float(np.sqrt(np.sum((a - b) ** 2)))


def precision_score(y_true, y_predicted):
        """precision score classification metric
        precision = TP / (TP + FP)
        where TP is true positives and FP is false positives
        """
    
        y_true = np.asarray(y_true)
        y_predicted = np.asarray(y_predicted)

        if y_true.shape != y_predicted.shape:
            raise ValueError("y_true and y_predicted must have the same shape")
        
        tp = np.sum((y_true == 1) & (y_predicted == 1))
        fp = np.sum((y_true == 0) & (y_predicted == 1))
        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))


def recall_score(y_true, y_predicted):
        """recall score classification metric
        recall = TP/ (TP+FN)  """
        y_true = np.asarray(y_true)
        y_predicted = np.asarray(y_predicted)
        if y_true.shape != y_predicted.shape:
            raise ValueError("y_true and y_predicted must have the same shape")
        
        tp = np.sum((y_true == 1) & (y_predicted == 1))
        fn = np.sum((y_true == 1) & (y_predicted == 0))
        if tp+fn ==0:
             return 0.0
        return float(tp / (tp+fn )) 


def f1_score(y_true, y_predicted):
        """f1 score classification metric
        f1 = 2 * ((precision * recall ) / (precision + recall))"""
        y_true = np.asarray(y_true)
        y_predicted = np.asarray(y_predicted)
        if y_true.shape != y_predicted.shape:
            raise ValueError("y_true and y_predicted must have the same shape")
        
        prec = precision_score(y_true, y_predicted)
        rec = recall_score(y_true, y_predicted)

        if prec + rec == 0:
             return 0.0
        return float(2 * (prec * rec) / (prec + rec))




import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):
    #get unique labels
    labels = np.unique(np.concatenate((y_true, y_pred)))
    #set up confusion matrix with shape (n_classes, n_classes)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    # calc TP, TN, FP, FN
    for t, p in zip(y_true, y_pred):
        cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
    
    # plot confusion matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    #text in each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.show()
    return cm


def roc_auc_score(y_true, y_scores):
    """ROC AUC score classification metric"""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")
    
    #sort by scores
    desc_score_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_score_indices]

    #cumulate true positives and false positives
    tpr = np.cumsum(y_true_sorted) / np.sum(y_true) 
    #cumulative sum devided by total positives gives an array of true positive rates at each threshold
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true)#same here

    #calc AUC (area under the ROC curve) using trapezoidal rule (integrate TPR with respect to FPR)
    auc = np.trapezoid(tpr, fpr)
    return float(auc)#if auc >0.5 it's a good model else it's a bad or random model

