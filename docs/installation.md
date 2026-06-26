# Installation

MiniML only needs NumPy for the core package.

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

For plotting examples:

```bash
pip install -e ".[examples]"
python examples/plot_loss_curves.py
python examples/classification_boundary.py
python examples/regularization_paths.py
```

For the Gradio demo:

```bash
pip install -e ".[demo]"
python demo/app.py
```

## Dependencies

The core library depends on NumPy only. Extra packages such as matplotlib, scikit-learn, pandas, and gradio are optional and used only for examples, comparisons, or demos.
