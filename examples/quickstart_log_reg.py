import numpy as np
from miniml.linear_model.logistic_regression import logistic_regression
from miniml.metrics.classification import accuracy

# dataset (linearly separable)
X = np.array([[-2], [-1], [0], [1], [2]])
y = np.array([0, 0, 0, 1, 1])

#train
model = logistic_regression(learning_rate=0.1, epochs=200, verbose_every=50)
model.fit(X, y)
#predict
predictions = model.predict(X)
#evaluate
acc = accuracy(y, predictions)

print("Predictions:", predictions)
print(f"Accuracy:{acc:.4f}")
