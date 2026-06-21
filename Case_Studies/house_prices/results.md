# House Prices Regression Results

These results were produced by running `run_analysis.py` with the local MiniML regression models.

## Setup

- Dataset: `data.csv`
- Target: `price`
- Invalid targets removed: rows with `price == 0`
- Extreme targets removed: top 1% of remaining prices
- Split: 80% train, 20% test with `seed=42`
- Scaling: MiniML `StandardScaler`
- Target transform: `np.log1p(price)` for training, `np.expm1` for reporting

## Holdout Metrics

| Model | MAE | MSE | R2 | Adjusted R2 |
|---|---:|---:|---:|---:|
| Linear Regression | 78,744.93 | 16,558,760,765.04 | 0.8109 | 0.7772 |
| Ridge Regression | 78,747.03 | 16,555,680,544.56 | 0.8109 | 0.7773 |
| Lasso Regression | 78,743.37 | 16,552,975,586.13 | 0.8110 | 0.7773 |
| Elastic Net | 78,745.06 | 16,557,388,274.47 | 0.8109 | 0.7773 |
| KNN Regressor | 95,440.77 | 23,772,796,411.93 | 0.7285 | 0.6802 |

## Best Model

`LassoRegression` is the best holdout model by R2, although the four linear-family models are extremely close. This makes sense because the log-price target and location features create a mostly linear signal after scaling and one-hot encoding.

## Visualizations

The script saves:

- `house_price_analysis.png`
- `house_price_loss_curves.png`

The main diagnostic plot shows that the model captures the central price range well. The residual plot is still wider for expensive homes, which is expected for real estate data because location, property quality, and unusual high-end sales are difficult to fully represent with the available tabular features.

## Notes

Cross-validation was intentionally not included in the final script. The local gradient-descent regressors are useful for learning and demonstration, but repeated fold refits make this case study slower without adding much value beyond the fixed holdout comparison.
