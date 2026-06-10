import numpy as np
from miniml.optim.gradient_descent import gradient_descent
from miniml.metrics.regression import MSE, MSE_gradient

class logistic_regression:
    def __init__(self, learning_rate=0.01, epochs=1000, verbose_every=None):
        self.learning_rate = learning_rate
        self.epoch = epochs
        self.verbose_every = verbose_every
        self.loss_history = []

        self.m