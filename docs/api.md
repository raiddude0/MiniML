# API Reference

MiniML estimators follow a small `fit` and `predict` interface.

## LinearRegression

```python
from miniml.linear_model import LinearRegression

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `learning_rate` | `float` | Step size for gradient descent. |
| `epochs` | `int` | Number of optimization iterations. |
| `verbose_every` | `int` or `None` | Print loss every N epochs. |

Attributes after fitting:

| Attribute | Description |
| --- | --- |
| `m` | Learned weights. |
| `b` | Learned bias term. |
| `loss_history` | Loss value recorded at each epoch. |

## RidgeRegression

```python
from miniml.linear_model import RidgeRegression

model = RidgeRegression(learning_rate=0.01, epochs=1000, alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `learning_rate` | `float` | Step size for gradient descent. |
| `epochs` | `int` | Number of optimization iterations. |
| `alpha` | `float` | Strength of the L2 penalty. |
| `verbose_every` | `int` or `None` | Print loss every N epochs. |

## LassoRegression

```python
from miniml.linear_model import LassoRegression

model = LassoRegression(learning_rate=0.01, epochs=1000, alpha=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `learning_rate` | `float` | Step size for gradient descent. |
| `epochs` | `int` | Number of optimization iterations. |
| `alpha` | `float` | Strength of the L1 penalty. |
| `verbose_every` | `int` or `None` | Print loss every N epochs. |

## LogisticRegression

```python
from miniml.linear_model import LogisticRegression

model = LogisticRegression(learning_rate=0.1, epochs=500)
model.fit(X_train, y_train)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `learning_rate` | `float` | Step size for gradient descent. |
| `epochs` | `int` | Number of optimization iterations. |
| `verbose_every` | `int` or `None` | Print loss every N epochs. |

## Metrics

```python
from miniml.metrics.regression import MSE, MAE, R2_score
from miniml.metrics.classification import accuracy, f1_score
```

Metrics are plain functions. They do not store model state.

## Preprocessing

```python
from miniml.preprocessing.standard_scaler import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Preprocessors use `fit`, `transform`, and `fit_transform`, which keeps training statistics separate from test data.
