import nnumpy as np

class GradientDescent:
    def __init__(self, learning_rate = 0.01, epochs = 1000, verbose_every = None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose_every = verbose_every

    def fit(self, X, y, model):
        for epoch in range(self.epochs):
            y_pred = model.forward(X)
            loss = model.loss(y, y_pred)
            model.backward(y, y_pred)
            model.update(self.learning_rate)

            if self.verbose_every is not None and epoch % self.verbose_every == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
                
        