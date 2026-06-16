
import numpy as np
from miniml.metrics.classification import euclidean_distance
from miniml.core.validation import check_X_y
from collections import defaultdict

class KNNRegressor:
    def __init__(self, n_neighbors=3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X, y = check_X_y(X, y)
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        distances = []
        
       
        for xi, yi in zip(self.X_train, self.y_train):
            dist = euclidean_distance(X, xi)
            distances.append((dist, yi))
        
        
        k_nearest = sorted(distances, key=lambda x: x[0])[:self.n_neighbors]
        
        
        # weighted average of neighbor values
        weights = [1 / (dist + 1e-9) for dist, _ in k_nearest] #inverted distances as weights, higher weight = closer neighbor 
        values = [val for _, val in k_nearest]
        result = np.average(values, weights=weights)
        return result

    def _check_is_fitted(self):
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("This model is not fitted yet. Call fit() before predict().")
        