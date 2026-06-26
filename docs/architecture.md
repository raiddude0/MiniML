# Architecture

MiniML is organized to keep the learning path clear.

## Package Layout

```text
miniml/
  linear_model/      linear and logistic estimators
  neighbors/         KNN estimators
  metrics/           regression and classification metrics
  model_selection/   train/test split, cross validation, grid search
  preprocessing/     scalers and feature transforms
  optim/             gradient descent
  core/              validation and base classes
```

`examples/` contains small scripts that show the library in action. `Case_Studies/` contains larger walkthroughs. `docs/` explains usage, architecture, and math.

## Estimator API

Models use:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

`fit` learns parameters from training data. `predict` uses those learned parameters on new data. This mirrors the common estimator pattern used by many Python ML tools, so switching between MiniML and other libraries feels familiar.

## Why Fit And Predict

Separating `fit` from `predict` prevents accidental retraining. It also makes the workflow easy to read:

1. Prepare data.
2. Fit the model.
3. Predict on new data.
4. Evaluate predictions.

## Metrics Are Separate From Models

Metrics live in `miniml.metrics` instead of inside each model. This keeps models focused on learning parameters and making predictions. It also lets the same metric work with any model that produces compatible predictions.

## Preprocessing In Workflows

Preprocessing classes such as `StandardScaler` are used before fitting a model:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
```

The scaler learns statistics from training data only. The test data is transformed with the same statistics, which avoids leaking information from the test set into training.

## Why NumPy Is The Only Core Dependency

The core package uses NumPy because it is enough to express the main ideas: arrays, vectorized operations, losses, gradients, and matrix multiplication. Keeping the dependency list small makes the code easier to inspect and easier to run in simple environments.
