# examples/quickstart_parity_demo.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SkLinear, Ridge as SkRidge, LogisticRegression as SkLogistic
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.datasets import make_classification

from miniml.linear_model import LinearRegression as MiniLinear, RidgeRegression as MiniRidge, LogisticRegression as MiniLogistic
from miniml.preprocessing.standard_scaler import StandardScaler as MiniScaler


# --- Linear Regression Comparison ---
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = 3 * X.flatten() + 2

mini_lin = MiniLinear().fit(X, y)
sk_lin = SkLinear().fit(X, y)

print("\n--- Linear Regression ---")
print("MiniML slope:", mini_lin.m, "intercept:", mini_lin.b)
print("Sklearn coef:", sk_lin.coef_, "intercept:", sk_lin.intercept_)

plt.scatter(X, y, label="Data")
plt.plot(X, mini_lin.predict(X), color="red", label="MiniML")
plt.plot(X, sk_lin.predict(X), color="blue", linestyle="--", label="Sklearn")
plt.legend()
plt.title("Linear Regression Comparison")
plt.show()


# --- Ridge Regression Comparison ---
mini_ridge = MiniRidge(alpha=10.0).fit(X, y)
sk_ridge = SkRidge(alpha=10.0).fit(X, y)

print("\n--- Ridge Regression (alpha=10) ---")
print("MiniML slope:", mini_ridge.m, "intercept:", mini_ridge.b)
print("Sklearn coef:", sk_ridge.coef_, "intercept:", sk_ridge.intercept_)

plt.scatter(X, y, label="Data")
plt.plot(X, mini_ridge.predict(X), color="green", label="MiniML Ridge")
plt.plot(X, sk_ridge.predict(X), color="orange", linestyle="--", label="Sklearn Ridge")
plt.legend()
plt.title("Ridge Regression Comparison")
plt.show()


# --- Logistic Regression Comparison ---

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)


mini = MiniLogistic(learning_rate=0.1, epochs=1000)
mini.fit(X, y)
preds_mini = mini.predict(X)
acc_mini = np.mean(preds_mini == y)


skl = SkLogistic(max_iter=1000)
skl.fit(X, y)
preds_skl = skl.predict(X)
acc_skl = np.mean(preds_skl == y)

print(f"MiniML Logistic Regression Accuracy: {acc_mini:.3f}")
print(f"scikit-learn Logistic Regression Accuracy: {acc_skl:.3f}")


def plot_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap=plt.cm.coolwarm,
        edgecolor="k", s=60
        )
    plt.title(title)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plot_boundary(mini, X, y, "MiniML Logistic Regression")

plt.subplot(1, 2, 2)
plot_boundary(skl, X, y, "scikit-learn Logistic Regression")

plt.tight_layout()
plt.show()

# --- StandardScaler Comparison ---
X_scale = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

mini_scaler = MiniScaler().fit(X_scale)
sk_scaler = SkScaler().fit(X_scale)

print("\n--- StandardScaler ---")
print("MiniML scaled:\n", mini_scaler.transform(X_scale))
print("Sklearn scaled:\n", sk_scaler.transform(X_scale))
