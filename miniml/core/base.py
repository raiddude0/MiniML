from typing import Protocol, Any

class BaseEstimator(Protocol):
    def fit(self, X, y) -> "BaseEstimator":
        ...

    def predict(self, X) -> Any:
        ...


class BaseTransformer(Protocol):
    def fit(self, X, y=None) -> "BaseTransformer":
        ...

    def transform(self, X) -> Any:
        ...

    def fit_transform(self, X, y=None) -> Any:
        ...

class BaseModel:
    """base class for all estimators/transformers."""

    def set_params(self, **params):
        """update model hyperparameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_params(self):
        """return current hyperparameters as a dict."""
        return {
            key: getattr(self, key)
            for key in self.__dict__.keys()
        }
