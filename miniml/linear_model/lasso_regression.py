
import numpy as np
from miniml.metrics.regression import MSE, lasso_loss, lasso_gradient
from miniml.core.validation import check_X_y, check_learning_rate, check_epochs

class LassoRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0, verbose_every=None):
        
        learning_rate = check_learning_rate(learning_rate)
        epochs = check_epochs(epochs)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
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

        
        for epoch in range(self.epochs):
            y_pred = self.m @ X.T + self.b
            loss = lasso_loss(y, y_pred, self.m, self.alpha)
            self.loss_history.append(loss)

            
            m_grad, b_grad = lasso_gradient(y, y_pred, X, self.m, self.alpha)

            
            self.m -= self.learning_rate * m_grad
            self.b -= self.learning_rate * b_grad

            
            if self.verbose_every is not None and epoch % self.verbose_every == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return self

    def _check_is_fitted(self):
        if self.m is None or self.b is None:
            raise ValueError("This model is not fitted yet. Call fit() before predict().")

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        return X @ self.m + self.b