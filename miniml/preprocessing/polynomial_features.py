
import numpy as np
from miniml.core.validation import check_X_y

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=False, include_interaction=False):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.include_bias = include_bias
        self.include_interaction = include_interaction

    def fit(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self
    
    def transform(self, X):
        """this transformer creates new features from the original features in the dataset"""
        X = np.asarray(X)
        N, n_features = X.shape
        features = [X]
        if self.include_bias:
            features.append(np.ones((N, 1)))
        
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                #slice each column and raise it to the power of d, then append to features list
                features.append(X[:, i:i+1] ** d)
            
        if self.include_interaction:
            for i in range(n_features):
                #loop to prevent doubles such as x1*x2 and x2*x1
                for j in range(i+1, n_features):
                    features.append((X[:, i] * X[:, j])[:, np.newaxis]) #reshape using np.newaxis so we can use np.hstack later on
        return np.hstack(features)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    