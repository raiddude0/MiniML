import numpy as np
from miniml.optim.gradient_descent import gradient_descent
from miniml.metrics.regression import MSE, MSE_gradient

class RidgeRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0, verbose_every = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
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

        #didn't use gradient descent class because i wanted to show the penalty for regularization:
        for i in range(self.epochs):
            y_predicted = X @ self.m + self.b
            loss = MSE(y, y_predicted) + self.alpha * np.sum(self.m **2) #L2 penalty for ridge regression
            self.loss_history.append(loss)
            m_gradient, b_gradient = MSE_gradient(X, y, y_predicted)

            self.m -= self.learning_rate * (m_gradient + 2 * self.alpha * self.m)
            self.b -= self.learning_rate * b_gradient


            if self.verbose_every is not None and i % self.verbose_every == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return self
    
    
    
            

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        return X @ self.m + self.b


    def _check_is_fitted(self):
        if self.m is None or self.b is None:
            raise ValueError("This model is not fitted yet. Call fit() before predict().")