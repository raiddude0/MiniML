# Breast Cancer Classification Case Study

This case study uses the local MiniML classification models to classify tumors as benign or malignant from the uploaded `breast-cancer.csv` dataset.

## Goal

Predict the `diagnosis` column:

- `B`: benign, encoded as `0`
- `M`: malignant, encoded as `1`

The focus is to evaluate local classification models and metrics from MiniML, without using external machine learning libraries.

## Dataset

The dataset contains 569 samples with 30 numeric cell-nucleus measurements, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal-dimension features.

The script:

- Loads `breast-cancer.csv`
- Drops the `id` column
- Removes missing rows if any exist
- Encodes `diagnosis` as a binary target
- Standardizes numeric features with MiniML's `StandardScaler`

## Models Used

All models for classification are from MiniML:

- `LogisticRegression`
- `KNNClassifier`

The split uses MiniML's `train_test_split`, and the reported metrics come from MiniML's local classification metrics.

## Metrics

The script reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC

## Outputs

Run the case study with:

```bash
python Case_Studies/breast_cancer_classification/run_analysis.py
```

The script prints metrics and saves:

- `figures/breast_cancer_analysis.png`: model comparison, confusion matrix, ROC curve, and logistic loss curve.
- `figures/breast_cancer_loss_curve_3d.png`: 3D logistic regression loss curve.

## Current Result

The best model by F1 score is currently `LogisticRegression`, with an F1 score of about `0.9444` and ROC AUC of about `0.9930`.
