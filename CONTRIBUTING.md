# Contributing To MiniML

Thanks for helping improve MiniML. This project is meant to stay simple, readable, and educational.

## Development Setup

Clone the project, then install it in editable mode:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Optional extras:

```bash
pip install -e ".[examples]"
pip install -e ".[demo]"
```

## Project Style

- Keep implementations small and easy to read.
- Prefer NumPy and simple Python over extra dependencies.
- Keep model logic separate from metrics, preprocessing, and model selection.
- Use clear names for learned parameters and public attributes.
- Add comments only when they explain non-obvious math or behavior.
- Avoid large abstractions unless they remove real duplication.

## Adding Or Changing Models

Models should follow the existing estimator pattern:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

When a model is trained with iterative optimization, store training losses in `loss_history` when possible. Prediction before fitting should raise a clear `ValueError`.

## Adding Metrics

Metrics should be plain functions in `miniml.metrics`. They should accept NumPy-like inputs, convert them with `np.asarray`, and validate shape where needed.

## Adding Examples

Examples should be runnable from the project root:

```bash
python examples/example_name.py
```

Prefer generated toy datasets unless the example is specifically a case study. If an example creates a figure, save it to `assets/` or the relevant case-study `figures/` directory.

## Adding Tests

Tests live in `tests/` and should stay structurally simple:

- Use small NumPy arrays.
- Test one behavior per test when possible.
- Check numerical values with `np.allclose` or `pytest.approx`.
- Add regression tests for bug fixes.
- Run `pytest` before submitting changes.

## Documentation

Update docs when behavior, examples, or public APIs change. Keep documentation practical:

- Show runnable code.
- Explain parameters directly.
- Include math only when it helps understanding.
- Link examples and generated figures when useful.

## Pull Request Checklist

Before opening a pull request:

- Tests pass with `pytest`.
- New behavior has focused tests.
- Public API changes are documented.
- Examples still run from the project root.
- The core dependency list remains small unless there is a strong reason to change it.
