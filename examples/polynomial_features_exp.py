import numpy as np
import matplotlib.pyplot as plt

from miniml.preprocessing.polynomial_features import PolynomialFeatures
from miniml.linear_model.linear_regression import LinearRegression  

# synthetic dataset
X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = 0.5 * X[:, 0]**2 + 2 * X[:, 0] + 3 + np.random.randn(30) * 0.5

# use polynomial features 
poly = PolynomialFeatures(degree=2, include_bias=True, include_interaction=False)
X_poly = poly.fit_transform(X)

# fit
model = LinearRegression()
model.fit(X_poly, y)

#predict
y_pred = model.predict(X_poly)

#results
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_pred, color="red", label="Polynomial Regression")
plt.legend()
plt.title("Polynomial Regression Example")
plt.show()

print("First 5 predictions:", y_pred[:5])
