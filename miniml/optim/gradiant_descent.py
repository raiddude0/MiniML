import nnumpy as np

class GradientDescent:
    def __init__(self, learning_rate = 0.01, epochs = 1000, verbose_every = None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose_every = verbose_every

        