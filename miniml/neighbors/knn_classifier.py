import numpy as np
from miniml.core.base import BaseModel
from miniml.metrics.classification import euclidean_distance
from miniml.core.validation import check_X_y_classification
from collections import defaultdict

class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors=3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X, y = check_X_y_classification(X, y)
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
        
       #tie handling
        votes = defaultdict(float)
        for dist, label in k_nearest:
            votes[label] += 1 / (dist + 1e-9)  # avoid division by zero

        result = max(votes, key=votes.get)
        return result

    def _check_is_fitted(self):
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("This model is not fitted yet. Call fit() before predict().")
        