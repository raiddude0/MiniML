# Linear Regression

Linear regression predicts a continuous target with a weighted sum of input features.

```text
y_hat = Xw + b
```

## Objective Function

Linear regression minimizes mean squared error:

```text
J(w, b) = (1 / n) * sum((y_hat - y)^2)
```

## Gradient

For predictions `y_hat = Xw + b`:

```text
dJ/dw = (2 / n) * X.T @ (y_hat - y)
dJ/db = (2 / n) * sum(y_hat - y)
```

Gradient descent updates the parameters:

```text
w = w - learning_rate * dJ/dw
b = b - learning_rate * dJ/db
```

The loss should usually move downward as training progresses:

![Linear regression loss curve](../../assets/loss_curve.png)

## Role Of Each Parameter

- `w`: feature weights learned from data.
- `b`: bias term, also called intercept.
- `learning_rate`: size of each gradient descent update.
- `epochs`: number of optimization steps.

## Common Failure Modes

- Learning rate is too high, causing the loss to explode.
- Learning rate is too low, causing very slow training.
- Features have very different scales.
- The relationship is not close to linear.
- Outliers dominate the squared error.

## When To Use It

Use linear regression for simple continuous prediction problems where the relationship between features and target is roughly linear.
