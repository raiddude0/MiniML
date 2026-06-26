from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from miniml.linear_model import LassoRegression, RidgeRegression
from miniml.preprocessing.standard_scaler import StandardScaler


def fit_coefficients(model_class, X, y, alphas):
    coefficients = []
    for alpha in alphas:
        model = model_class(learning_rate=0.03, epochs=600, alpha=alpha)
        model.fit(X, y)
        coefficients.append(np.abs(model.m))
    return np.array(coefficients)


def main():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(180, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.3, size=180)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])

    ridge_coefficients = fit_coefficients(RidgeRegression, X_scaled, y, alphas)
    lasso_coefficients = fit_coefficients(LassoRegression, X_scaled, y, alphas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for feature_idx in range(X.shape[1]):
        axes[0].plot(alphas, ridge_coefficients[:, feature_idx], marker="o", label=f"w{feature_idx}")
        axes[1].plot(alphas, lasso_coefficients[:, feature_idx], marker="o", label=f"w{feature_idx}")

    axes[0].set_title("Ridge")
    axes[1].set_title("Lasso")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("absolute coefficient value")
    fig.suptitle("Regularization Paths")

    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "regularization_paths.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
