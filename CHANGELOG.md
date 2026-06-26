# Changelog

All notable changes to MiniML are documented here.

The format is inspired by Keep a Changelog, and this project uses simple semantic versioning while it is under active development.

## [0.1.0] - 2026-06-26

### Added

- Linear regression with gradient descent and loss history.
- Ridge, Lasso, and Elastic Net regression.
- Logistic regression with `predict` and `predict_proba`.
- KNN classifier and regressor.
- Regression metrics: MSE, MAE, RMSE, R2, adjusted R2.
- Classification metrics: accuracy, precision, recall, F1, confusion matrix, ROC AUC.
- Preprocessing tools: standard scaler, min-max scaler, polynomial features.
- Model selection tools: train/test split, cross validation, grid search.
- Example scripts for quickstarts, visualizations, and scikit-learn comparison.
- Visual assets for loss curves, classification boundary, and regularization paths.
- Documentation for installation, API usage, architecture, tutorials, and math notes.
- Gradio demo using MiniML linear regression.
- Test suite for core models, metrics, preprocessing, model selection, and KNN.

### Fixed

- KNN regressor now handles single-sample prediction inputs such as `[1]`.

### Notes

- MiniML is intended as an educational project. The implementation favors readability and explicit math over production-level solver performance.
