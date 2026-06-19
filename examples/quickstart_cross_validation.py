import numpy as np
from miniml.linear_model.linear_regression import LinearRegression
from miniml.metrics.regression import MSE
from miniml.model_selection.cross_validation import cross_val_score

# y = 2x + 1
X = np.arange(10).reshape(-1, 1)
y = 2 * X.flatten() + 1


metric = MSE
model = LinearRegression()


scores = cross_val_score(model, X, y, k=5, metric=metric, seed=0)

print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("Std deviation:", scores.std())
