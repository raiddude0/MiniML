import numpy as np
from miniml.linear_model.ridge_regression import RidgeRegression as Ridge
from miniml.metrics.regression import R2_score
from miniml.model_selection.grid_search import grid_search

# Perfectly linear dataset
X = np.array([[0], [1], [2], [3], [4]])
y = 2 * X.flatten()  # y = 2x

# Candidate hyperparameters
param_grid = [
    {"alpha": 0},     # no penalty, should fit perfectly
    {"alpha": 0.1},   # light penalty
    {"alpha": 10.0}   # heavy penalty
]

# Run grid search
model = Ridge(learning_rate=0.001, epochs=5000)
best_params, best_score = grid_search(model, param_grid, X, y, metric=R2_score, k=2, seed=0, greater_is_better=True)

print("Grid Search Results:")
for p in param_grid:
    model.set_params(**p)
    scores = grid_search(model, [p], X, y, metric=R2_score, k=2, seed=0, greater_is_better=True)[1]
    print(f"Params={p}, Mean R2 Score={scores:.6f}")

print("\nBest Params:", best_params)
print("Best Mean R2 Score:", best_score)