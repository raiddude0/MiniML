import numpy as np
import matplotlib.pyplot as plt
from miniml.neighbors.knn_regressor import KNNRegressor
from miniml.metrics.regression import MAE


X_train = np.array([[0],[1],[2],[3],[4],[5]])
y_train = np.array([0,2,4,6,8,10])


knn = KNNRegressor(n_neighbors=2)
knn.fit(X_train, y_train)


X_test = np.array([[1.5],[2.5],[3.5]])
y_true = np.array([3,5,7])
y_pred = [knn.predict(x) for x in X_test]

print("Mean Absolute Error:", MAE(y_true, y_pred))


X_line = np.linspace(0,5,50).reshape(-1,1)
y_line = [knn.predict(x) for x in X_line]

plt.scatter(X_train, y_train, color="blue", label="Train")
plt.plot(X_line, y_line, color="red", label="KNN Predictions")
plt.title("KNN Regression Example")
plt.legend()
plt.show()
