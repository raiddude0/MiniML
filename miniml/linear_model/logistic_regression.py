import numpy as np
from miniml.optim.gradient_descent import gradient_descent
from miniml.metrics.classification import log_loss, log_loss_gradient

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, verbose_every=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose_every = verbose_every
        self.loss_history = []

        self.m = None
        self.b = None
        
    
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        N, n_features = X.shape

        self.m = np.zeros(n_features)
        self.b = 0
        for i in range(self.epochs):
            loss = log_loss(X, y, self.m, self.b)
            self.loss_history.append(loss)
            m_gradient, b_gradient = log_loss_gradient(X, y, self.m, self.b)
            self.m -= self.learning_rate * m_gradient
            self.b -= self.learning_rate * b_gradient

            if self.verbose_every is not None and i % self.verbose_every == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)
        z = X @ self.m + self.b
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X, threshold=0.5):
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    

    def _check_is_fitted(self):
        if self.m is None or self.b is None:
            raise ValueError("This model is not fitted yet. Call fit() before predict().")