# Gradient Descent

Gradient descent is an iterative optimization method. It changes model parameters in the direction that reduces the loss.

## Objective Function

Gradient descent does not define one objective by itself. It optimizes the objective supplied by a model, such as mean squared error for linear regression or log loss for logistic regression.

```text
minimize J(parameters)
```

## Gradient

At each epoch, compute the gradient of the objective:

```text
gradient = dJ / dparameters
parameters = parameters - learning_rate * gradient
```

## Role Of Each Parameter

- `parameters`: the values being learned, usually weights and bias.
- `learning_rate`: how far to move on each update.
- `epochs`: how many updates to run.
- `loss_history`: recorded loss values used to inspect training.

## Common Failure Modes

- Learning rate is too high and the loss increases.
- Learning rate is too low and training barely moves.
- Too few epochs leave the model undertrained.
- Poor feature scaling makes updates uneven.
- Noisy data can make the loss curve bounce.

## When To Use It

Use gradient descent when a model has a differentiable objective and you want a simple optimization method that is easy to understand and implement.
