# Breast Cancer Classification Results

These results were produced by running `run_analysis.py` with the local MiniML classification models and metrics.

## Setup

- Dataset: `breast-cancer.csv`
- Target: `diagnosis`
- Encoding: `B = 0`, `M = 1`
- Dropped column: `id`
- Split: 80% train, 20% test with `seed=42`
- Scaling: MiniML `StandardScaler`

## Holdout Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9646 | 1.0000 | 0.8947 | 0.9444 | 0.9930 |
| KNN Classifier | 0.9558 | 0.9714 | 0.8947 | 0.9315 | 0.9095 |

## Best Model

`LogisticRegression` is the best model by F1 score. It also has the strongest ROC AUC, which suggests its probability scores rank malignant and benign cases very well.

## Interpretation

The logistic model reaches perfect precision on the holdout split, meaning every predicted malignant case was truly malignant in this run. Its recall is lower than precision, so it still misses a few malignant cases. In a medical screening setting, that tradeoff would matter: a lower decision threshold could improve recall, but may also increase false positives.

The KNN classifier is close on accuracy and recall, but it has weaker ROC AUC because this script uses its hard class predictions as scores. Logistic regression is better suited for ROC analysis because it exposes `predict_proba`.

## Visualizations

The script saves:

- `figures/breast_cancer_analysis.png`
- `figures/breast_cancer_loss_curve_3d.png`

The first figure includes metric comparison, confusion matrix, ROC curve, and logistic loss. The second figure shows the logistic training loss in 3D.
