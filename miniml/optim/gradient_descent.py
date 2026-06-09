import numpy as np

class gradient_descent:
    def __init__(self, learning_rate=0.01, epochs=1000, verbose_every=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose_every = verbose_every
        self.loss_history = []

    def fit(self, X, y, loss_f, grad_f, params):
        """
        Run batch gradient descent.

        Parameters
        ----------
        X : 
            Input features
        y : 
            Target values
        loss_f : callable
            Function loss_f(X, y, params) -> float
        grad_f : callable
            Function grad_f(X, y, params) -> np.ndarray
        params : np.ndarray
            Initial parameters

        Returns
        -------
        np.ndarray
            Optimized parameters
        """
        self.params = params

        for epoch in range(self.epochs):
            loss = loss_f(X, y, self.params)
            self.loss_history.append(loss)

            grad = grad_f(X, y, self.params)
            self.params -= self.learning_rate * grad

            if self.verbose_every is not None and epoch % self.verbose_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return self.params
