from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from miniml.linear_model import LogisticRegression
from miniml.preprocessing.standard_scaler import StandardScaler


def main():
    rng = np.random.default_rng(7)
    class_0 = rng.normal(loc=(-1.5, -1.0), scale=0.7, size=(80, 2))
    class_1 = rng.normal(loc=(1.5, 1.0), scale=0.7, size=(80, 2))

    X = np.vstack([class_0, class_1])
    y = np.array([0] * len(class_0) + [1] * len(class_1))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(learning_rate=0.1, epochs=500)
    model.fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probabilities, levels=20, cmap="RdBu", alpha=0.35)
    plt.contour(xx, yy, probabilities, levels=[0.5], colors="black")
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap="RdBu", edgecolor="white")
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")

    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "classification_boundary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
