import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from miniml.preprocessing.standard_scaler import StandardScaler
from miniml.linear_model import LogisticRegression
from miniml.neighbors.knn_classifier import KNNClassifier
from miniml.model_selection.split import train_test_split
from miniml.metrics.classification import (
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


df = pd.read_csv(__file__.replace("run_analysis.py", "breast-cancer.csv"))
df = df.drop(columns=["id"], errors="ignore")
df = df.dropna().copy()

y = (df["diagnosis"] == "M").astype(int).values
X = df.drop("diagnosis", axis=1).values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(learning_rate=0.05, epochs=3000),
    "KNN Classifier": KNNClassifier(n_neighbors=5),
}

results = {}
predictions = {}
probabilities = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = np.array([model.predict(row) for row in X_test_scaled])
        y_proba = y_pred.astype(float)

    predictions[name] = y_pred
    probabilities[name] = y_proba
    results[name] = {
        "Accuracy": accuracy(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
    }

    print(
        f"{name} - "
        f"Accuracy: {results[name]['Accuracy']:.4f}, "
        f"Precision: {results[name]['Precision']:.4f}, "
        f"Recall: {results[name]['Recall']:.4f}, "
        f"F1: {results[name]['F1']:.4f}, "
        f"ROC AUC: {results[name]['ROC AUC']:.4f}"
    )

best_model_name = max(results, key=lambda model_name: results[model_name]["F1"])
best_predictions = predictions[best_model_name]

cm = np.zeros((2, 2), dtype=int)
for true_label, predicted_label in zip(y_test, best_predictions):
    cm[true_label, predicted_label] += 1

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
x = np.arange(len(metric_names))
bar_width = 0.35

for model_index, (name, metrics) in enumerate(results.items()):
    axes[0, 0].bar(
        x + model_index * bar_width,
        [metrics[metric] for metric in metric_names],
        width=bar_width,
        label=name,
    )

axes[0, 0].set_title("Classification metrics")
axes[0, 0].set_xticks(x + bar_width / 2)
axes[0, 0].set_xticklabels(metric_names, rotation=25)
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].legend()

im = axes[0, 1].imshow(cm, cmap="Blues")
axes[0, 1].set_title(f"Confusion matrix: {best_model_name}")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(["Benign", "Malignant"])
axes[0, 1].set_yticklabels(["Benign", "Malignant"])
fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0, 1].text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )

for name, scores in probabilities.items():
    thresholds = np.linspace(0, 1, 101)
    tpr = []
    fpr = []
    for threshold in thresholds:
        threshold_predictions = (scores >= threshold).astype(int)
        tp = np.sum((y_test == 1) & (threshold_predictions == 1))
        fp = np.sum((y_test == 0) & (threshold_predictions == 1))
        tn = np.sum((y_test == 0) & (threshold_predictions == 0))
        fn = np.sum((y_test == 1) & (threshold_predictions == 0))
        tpr.append(tp / (tp + fn) if tp + fn > 0 else 0)
        fpr.append(fp / (fp + tn) if fp + tn > 0 else 0)

    axes[1, 0].plot(fpr, tpr, label=f"{name} AUC={results[name]['ROC AUC']:.3f}")

axes[1, 0].plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
axes[1, 0].set_title("ROC curve")
axes[1, 0].set_xlabel("False positive rate")
axes[1, 0].set_ylabel("True positive rate")
axes[1, 0].legend()

logistic_model = trained_models["Logistic Regression"]
axes[1, 1].plot(logistic_model.loss_history, color="#E45756")
axes[1, 1].set_title("Logistic regression loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Log loss")

fig.tight_layout()
figure_path = __file__.replace("run_analysis.py", "figures\\breast_cancer_analysis.png")
plt.savefig(figure_path, dpi=150)
plt.close(fig)

loss_fig = plt.figure(figsize=(11, 8))
ax = loss_fig.add_subplot(111, projection="3d")
losses = np.asarray(logistic_model.loss_history, dtype=float)
epochs = np.arange(1, len(losses) + 1)
learning_rate_axis = np.full_like(epochs, logistic_model.learning_rate, dtype=float)
ax.plot(epochs, learning_rate_axis, losses, color="#E45756", linewidth=2)
ax.set_title("3D logistic regression loss curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning rate")
ax.set_zlabel("Log loss")
ax.view_init(elev=25, azim=-55)

loss_fig.tight_layout()
loss_figure_path = __file__.replace("run_analysis.py", "figures\\breast_cancer_loss_curve_3d.png")
plt.savefig(loss_figure_path, dpi=150)
plt.close(loss_fig)

print(f"\nBest model by F1: {best_model_name}")
print(f"Saved visualization to: {figure_path}")
print(f"Saved 3D loss curve to: {loss_figure_path}")
