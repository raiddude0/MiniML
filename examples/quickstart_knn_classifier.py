import numpy as np
import matplotlib.pyplot as plt
from miniml.neighbors.knn_classifier import KNNClassifier
from miniml.metrics.classification import accuracy 


X_train = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y_train = np.array(["A","A","A","B","B","B"])


knn = KNNClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


X_test = np.array([[2,2],[7,6],[5,5]])
y_true = np.array(["A","B","B"])
y_pred = [knn.predict(x) for x in X_test]

print("Accuracy:", accuracy(y_true, y_pred))


plt.scatter(X_train[:,0], X_train[:,1], c=(y_train=="A"), cmap="coolwarm", label="Train")

plt.scatter(X_test[:,0], X_test[:,1], c="green", marker="x", s=100, label="Test")
plt.title("KNN Classification Example")
plt.legend()
plt.show()
