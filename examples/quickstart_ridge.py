import numpy as np
from miniml.linear_model.ridge_regression import ridge_regression
from miniml.metrics.regression import MSE, RMSE, R2_score, MAE

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])
# Train ridge regression model
model = ridge_regression(learning_rate=0.01, epochs=1000, alpha=1.0, verbose_every=100)
model.fit(X, y)
# Make predictions
predictions = model.predict(X)
# Evaluate model
mse = MSE(y, predictions)
rmse = RMSE(y, predictions)
r2 = R2_score(y, predictions)
mae = MAE(y, predictions)

print("Predictions:", predictions)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

