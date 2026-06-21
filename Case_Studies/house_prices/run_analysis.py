import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from miniml.preprocessing.standard_scaler import StandardScaler
from miniml.linear_model import LinearRegression, RidgeRegression, LassoRegression, ElasticNet
from miniml.neighbors.knn_regressor import KNNRegressor
from miniml.model_selection.split import train_test_split
from miniml.metrics.regression import MAE, MSE, R2_score, adjusted_R2_score

df = pd.read_csv(__file__.replace("run_analysis.py", "data.csv"))

#remove impossible prices and the most extreme 1% of sales
df = df[df["price"] > 0].copy()
df = df[df["price"] <= df["price"].quantile(0.99)].copy()

df["date"] = pd.to_datetime(df["date"])
df["sale_month"] = df["date"].dt.month
df["house_age"] = df["date"].dt.year - df["yr_built"]
df["renovated"] = (df["yr_renovated"] > 0).astype(int)
df["years_since_renovation"] = np.where(
    df["yr_renovated"] > 0,
    df["date"].dt.year - df["yr_renovated"],
    df["house_age"],
)
df["sqft_living_per_bedroom"] = df["sqft_living"] / np.maximum(df["bedrooms"], 1)
df["sqft_lot_log"] = np.log1p(df["sqft_lot"])

features = df.drop(["price", "date", "street", "country"], axis=1)
features = pd.get_dummies(features, columns=["city", "statezip"], drop_first=True, dtype=float)

X = features.values.astype(float)
y = np.log1p(df["price"].values.astype(float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

# Some one-hot categories can be absent from the training split; dropping
# constant columns keeps StandardScaler stable.
keep_columns = X_train.std(axis=0) > 0
X_train = X_train[:, keep_columns]
X_test = X_test[:, keep_columns]

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_configs = {
    "Linear Regression": lambda: LinearRegression(learning_rate=0.01, epochs=800),
    "Ridge Regression": lambda: RidgeRegression(learning_rate=0.01, epochs=800, alpha=0.001),
    "Lasso Regression": lambda: LassoRegression(learning_rate=0.01, epochs=800, alpha=0.0001),
    "Elastic Net": lambda: ElasticNet(learning_rate=0.01, epochs=800, alpha=0.0001, l1_ratio=0.2),
    "KNN Regressor": lambda: KNNRegressor(n_neighbors=5),
}
results = {}
predictions = {}
trained_models = {}
y_test_price = np.expm1(y_test)

for name, build_model in model_configs.items():
    model = build_model()
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    preds = np.expm1(model.predict(X_test_scaled))
    predictions[name] = preds
    results[name] = {
        "MAE": MAE(y_test_price, preds),
        "MSE": MSE(y_test_price, preds),
        "R2": R2_score(y_test_price, preds),
        "Adjusted R2": adjusted_R2_score(y_test_price, preds, X_train_scaled.shape[1]),
    }
    print(
        f"{name} - "
        f"MAE: {results[name]['MAE']:.2f}, "
        f"MSE: {results[name]['MSE']:.2f}, "
        f"R2: {results[name]['R2']:.4f}, "
        f"Adjusted R2: {results[name]['Adjusted R2']:.4f}"
    )

best_model_name = max(results, key=lambda name: results[name]["R2"])
best_predictions = predictions[best_model_name]
residuals = y_test_price - best_predictions

#PLOTS
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].bar(results.keys(), [metrics["R2"] for metrics in results.values()], color="#4C78A8")
axes[0, 0].set_title("Holdout R2 by model")
axes[0, 0].set_ylabel("R2")
axes[0, 0].tick_params(axis="x", rotation=25)

axes[0, 1].scatter(y_test_price, best_predictions, alpha=0.55, color="#F58518", edgecolors="none")
limit = max(y_test_price.max(), best_predictions.max())
axes[0, 1].plot([0, limit], [0, limit], color="#333333", linewidth=1)
axes[0, 1].set_title(f"Actual vs predicted: {best_model_name}")
axes[0, 1].set_xlabel("Actual price")
axes[0, 1].set_ylabel("Predicted price")

axes[1, 0].scatter(best_predictions, residuals, alpha=0.55, color="#54A24B", edgecolors="none")
axes[1, 0].axhline(0, color="#333333", linewidth=1)
axes[1, 0].set_title("Residuals")
axes[1, 0].set_xlabel("Predicted price")
axes[1, 0].set_ylabel("Actual - predicted")

axes[1, 1].bar(results.keys(), [metrics["MAE"] for metrics in results.values()], color="#B279A2")
axes[1, 1].set_title("Holdout MAE by model")
axes[1, 1].set_ylabel("MAE")
axes[1, 1].tick_params(axis="x", rotation=25)

fig.tight_layout()
figure_path = __file__.replace("run_analysis.py", "house_price_analysis.png")
plt.savefig(figure_path, dpi=150)
plt.close(fig)

loss_histories = {
    name: np.asarray(model.loss_history, dtype=float)
    for name, model in trained_models.items()
    if hasattr(model, "loss_history") and len(model.loss_history) > 0
}

if loss_histories:
    loss_fig = plt.figure(figsize=(12, 8))
    ax = loss_fig.add_subplot(111, projection="3d")

    for model_index, (name, losses) in enumerate(loss_histories.items()):
        epochs = np.arange(1, len(losses) + 1)
        model_axis = np.full_like(epochs, model_index, dtype=float)
        ax.plot(epochs, model_axis, np.log10(losses + 1e-12), label=name, linewidth=2)

    ax.set_title("Training loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Model")
    ax.set_zlabel("Log10 loss")
    ax.set_yticks(range(len(loss_histories)))
    ax.set_yticklabels(loss_histories.keys())
    ax.view_init(elev=25, azim=-55)
    ax.legend(loc="upper right")

    loss_fig.tight_layout()
    loss_figure_path = __file__.replace("run_analysis.py", "house_price_loss_curves.png")
    plt.savefig(loss_figure_path, dpi=150)
    plt.close(loss_fig)

print(f"\nBest holdout model: {best_model_name}")
print(f"Saved visualization to: {figure_path}")
if loss_histories:
    print(f"Saved loss curve visualization to: {loss_figure_path}")


