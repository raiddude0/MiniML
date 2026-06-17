import numpy as np
from miniml.core.validation import check_X_y, check_learning_rate, check_epochs
from miniml.metrics.regression import elastic_net_loss, elastic_net_gradient
from miniml.optim.gradient_descent import gradient_descent

class ElasticNet:
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0, l1_ratio=0.5, verbose_every=None):
        learning_rate= check_learning_rate(learning_rate)
        epochs = check_epochs(epochs)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose_every = verbose_every
        self.loss_history = []

        self.m = None
        self.b = None
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        N, n_features = X.shape
        self.m = np.zeros(n_features)
        self.b = 0

        #we use gradient descent to get best parameters
        optimizer = gradient_descent(learning_rate=self.learning_rate, epochs=self.epochs, verbose_every=self.verbose_every)
        params = np.concatenate([self.m, [self.b]])
        loss_f = lambda X, y, m, b: elastic_net_loss(y, X @ m + b, m, self.alpha, self.l1_ratio)
        grad_f = lambda X, y, m, b: elastic_net_gradient(y, X @ m + b, X, m, self.alpha, self.l1_ratio)
        optimized_params = optimizer.fit(X, y, loss_f=loss_f, grad_f=grad_f, params=params)
        self.m, self.b = optimized_params[:-1], optimized_params[-1]
        self.loss_history = optimizer.loss_history
        
            
        return self
    

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        return X @ self.m + self.b
    
    def _check_is_fitted(self):
        if self.m is None or self.b is None:
            raise ValueError("This model is not fitted yet. Call fit() before predict().")
        
    
