# House Prices Regression Case Study

This case study uses the local MiniML regression models to predict house sale prices from the `data.csv` file. The analysis is implemented in `run_analysis.py` and intentionally uses the models, preprocessing tools, split helper, and metrics from this repository.

## Goal

Predict the final house price from numeric housing attributes and location information, then compare several MiniML regressors on the same holdout split.

## Dataset

The dataset contains 4,600 house records with sale price, home size, lot size, rooms, floors, condition, waterfront/view indicators, build/renovation years, street, city, state ZIP, and sale date.

The script performs a small amount of cleaning before modeling:

- Removes rows where `price == 0`, because they are not valid sale targets.
- Removes the top 1% of prices, because extreme luxury/outlier sales dominate squared-error metrics.
- Parses the sale date.
- Drops `street` and `country`.

## Feature Engineering

The analysis adds several features that help the local models capture the dataset better:

- `sale_month`
- `house_age`
- `renovated`
- `years_since_renovation`
- `sqft_living_per_bedroom`
- `sqft_lot_log`
- One-hot encoded `city` and `statezip`

The target is transformed with `np.log1p(price)` during training. Predictions are transformed back to dollar prices with `np.expm1` before computing metrics.

## Models Used

All models are from MiniML:

- `LinearRegression`
- `RidgeRegression`
- `LassoRegression`
- `ElasticNet`
- `KNNRegressor`

The features are standardized with MiniML's `StandardScaler`, and the holdout split uses MiniML's `train_test_split`.

## Outputs

Run the case study with:

```bash
python Case_Studies/house_prices/run_analysis.py
```

The script prints holdout metrics and saves:

- `house_price_analysis.png`: model comparison, actual vs predicted prices, residuals, and MAE comparison.
- `house_price_loss_curves.png`: 3D training loss curves for the gradient-descent regressors.

## Current Result

The best holdout model is currently `LassoRegression`, with an R2 of about `0.8110` on the cleaned holdout set.
