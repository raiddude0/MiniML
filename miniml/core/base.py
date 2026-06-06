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
