import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """Compute the mean and std of X for later scaling.
            if std is zero, raise an error because we cannot scale data with zero variance
        """
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        if np.any(self.scale_ == 0):
            raise ValueError("std is zero, cannot scale data")
        
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("call fit before transform")
        
        X = np.asarray(X)
        return(X - self.mean_) / self.scale_
    
    def inverse_transform(self, X_scaled):
        """
        Transform data back to its original scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("call fit before inverse_transform")
        
        X_scaled = np.asarray(X_scaled)

        return X_scaled * self.scale_ + self.mean_
    
    def fit_transform(self, X, y= None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)
    
