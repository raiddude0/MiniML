from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from miniml.linear_model import LinearRegression


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 1))
    y = 4 * X[:, 0] + 2 + rng.normal(scale=0.4, size=120)

    model = LinearRegression(learning_rate=0.05, epochs=200)
    model.fit(X, y)

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history)
    plt.title("Linear Regression Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.grid(alpha=0.3)

    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "loss_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
