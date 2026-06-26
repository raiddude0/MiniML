# Comparing With scikit-learn

This page shows a concrete comparison between MiniML `LinearRegression` and scikit-learn `LinearRegression` on the same data.

## Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

from miniml.linear_model import LinearRegression
from miniml.metrics.regression import MSE
from miniml.model_selection.split import train_test_split
from miniml.preprocessing.standard_scaler import StandardScaler


X = np.array([
    [800, 1],
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2200, 4],
    [2600, 4],
    [3000, 5],
], dtype=float)

y = np.array([
    180000,
    225000,
    250000,
    320000,
    355000,
    430000,
    490000,
    560000,
], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2, shuffle=False
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mini = LinearRegression(learning_rate=0.05, epochs=3000)
mini.fit(X_train_scaled, y_train)
mini_predictions = mini.predict(X_test_scaled)

sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X_train_scaled, y_train)
sklearn_predictions = sklearn_model.predict(X_test_scaled)

print("MiniML weights:", mini.m)
print("MiniML bias:", mini.b)
print("scikit-learn weights:", sklearn_model.coef_)
print("scikit-learn bias:", sklearn_model.intercept_)
print("MiniML predictions:", mini_predictions)
print("scikit-learn predictions:", sklearn_predictions)
print("MiniML MSE:", MSE(y_test, mini_predictions))
print("scikit-learn MSE:", mean_squared_error(y_test, sklearn_predictions))
```

Example output:

```text
MiniML weights: [83255.7 21774.3]
MiniML bias: 400833.3
scikit-learn weights: [83257.2 21772.8]
scikit-learn bias: 400833.3
MiniML predictions: [176434.9 225984.1]
scikit-learn predictions: [176435.7 225983.9]
MiniML MSE: 6839308.3
scikit-learn MSE: 6836028.9
```

The numbers are close because both models fit the same scaled training data. They are not exactly identical because MiniML reaches the solution through gradient descent, while scikit-learn solves linear regression directly.

## Why Scale The Features

The first feature is home size and the second feature is bedroom count. Their values are on very different scales, so MiniML gradient descent trains more smoothly after `StandardScaler`. The same scaled data is passed to scikit-learn so the comparison is fair.
