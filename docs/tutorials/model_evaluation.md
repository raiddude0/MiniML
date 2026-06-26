# Model Evaluation

Model evaluation checks how well predictions match the true target values.

## Regression

```python
from miniml.metrics.regression import MAE, MSE, RMSE, R2_score

predictions = model.predict(X_test)

print("MAE:", MAE(y_test, predictions))
print("MSE:", MSE(y_test, predictions))
print("RMSE:", RMSE(y_test, predictions))
print("R2:", R2_score(y_test, predictions))
```

Use regression metrics when the target is continuous, such as price, distance, or score.

## Classification

```python
from miniml.metrics.classification import accuracy, precision_score, recall_score, f1_score

labels = model.predict(X_test)

print("Accuracy:", accuracy(y_test, labels))
print("Precision:", precision_score(y_test, labels))
print("Recall:", recall_score(y_test, labels))
print("F1:", f1_score(y_test, labels))
```

Use classification metrics when the target is a class label.

## Train And Test Data

Always evaluate on data that was not used for fitting:

```python
from miniml.model_selection.split import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
```

This gives a more realistic estimate of how the model behaves on new examples.
