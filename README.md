# MiniML

MiniML is a small NumPy-based machine learning library implemented from scratch. It is designed for learning how common machine learning models work internally: losses, gradients, optimization, preprocessing, metrics, and model evaluation.

The goal is clarity, not replacing scikit-learn.

## Features

- Linear regression trained with batch gradient descent
- Ridge, Lasso, and Elastic Net regression
- Logistic regression for binary classification
- K-nearest neighbors classifier and regressor
- Standard scaling, min-max scaling, and polynomial features
- Train/test split, cross validation, and grid search
- Regression metrics: MSE, MAE, RMSE, R2, adjusted R2
- Classification metrics: accuracy, precision, recall, F1, confusion matrix, ROC AUC
- Loss history tracking for gradient-based models
- Example scripts, case studies, visualizations, and a small Gradio demo
- Tests for core behavior and scikit-learn parity checks

## Installation

From the project root:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

For plotting examples:

```bash
pip install -e ".[examples]"
```

For the Gradio demo:

```bash
pip install -e ".[demo]"
```

## Quickstart

```python
import numpy as np

from miniml.linear_model import LinearRegression
from miniml.metrics.regression import MSE

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

predictions = model.predict(np.array([[6], [7]]))

print(predictions)
print("training loss:", model.loss_history[-1])
print("mse:", MSE(np.array([15, 17]), predictions))
```

## Logistic Regression Example

```python
import numpy as np

from miniml.linear_model import LogisticRegression

X = np.array([[-2], [-1], [0], [1], [2]])
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression(learning_rate=0.1, epochs=300)
model.fit(X, y)

print(model.predict(np.array([[-0.5], [1.5]])))
print(model.predict_proba(np.array([[-0.5], [1.5]])))
```

## Visual Examples

The scripts in `examples/` create small generated datasets, train MiniML models, and save figures into `assets/`.

```bash
python examples/plot_loss_curves.py
python examples/classification_boundary.py
python examples/regularization_paths.py
```

### Loss Curve

![Linear regression loss curve](assets/loss_curve.png)

### Classification Boundary

![Logistic regression decision boundary](assets/classification_boundary.png)

### Regularization Paths

![Regularization paths](assets/regularization_paths.png)

## Project Structure

```text
miniml/
  linear_model/      linear, ridge, lasso, elastic net, logistic regression
  neighbors/         KNN classifier and regressor
  metrics/           regression and classification metrics
  model_selection/   train/test split, cross validation, grid search
  preprocessing/     scalers and polynomial features
  optim/             gradient descent
  core/              shared base classes and validation

examples/            quickstarts and visual examples
docs/                API, tutorials, architecture, and math notes
Case_Studies/        larger walkthroughs on real-style datasets
demo/                Gradio demo app
tests/               pytest test suite
```

## Documentation

Start here:

- [Documentation index](docs/index.md)
- [Installation](docs/installation.md)
- [API reference](docs/api.md)
- [Architecture](docs/architecture.md)
- [Getting started](docs/tutorials/getting_started.md)
- [Comparing with scikit-learn](docs/tutorials/comparing_with_sklearn.md)

Math notes:

- [Linear regression](docs/math/linear_regression.md)
- [Ridge regression](docs/math/ridge_regression.md)
- [Logistic regression](docs/math/logistic_regression.md)
- [Gradient descent](docs/math/gradient_descent.md)

## Demo

Run the small Gradio demo:

```bash
python demo/app.py
```

The demo trains a MiniML linear regression model on a generated house-price-style dataset and predicts price from size, bedrooms, and age.

## Tests

Run the full suite:

```bash
pytest
```

The tests cover model behavior, metrics, preprocessing, model selection, KNN, and comparisons with scikit-learn where useful.

## Design Goals

- Keep the core library simple and readable.
- Use NumPy as the only required runtime dependency.
- Prefer explicit math and direct implementations over hidden abstractions.
- Keep examples small enough to understand in one sitting.
- Separate models, metrics, preprocessing, and model selection into clear modules.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
