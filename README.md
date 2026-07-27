# Hybrid Demand Forecasting — Prophet + XGBoost

Retail demand forecasting on the Rossmann Store Sales dataset, comparing a Prophet baseline, an end-to-end XGBoost model, and a Prophet + XGBoost hybrid to test whether hybrid complexity is actually justified.

## Overview

The task combines two forecasting philosophies: **Prophet** captures the time structure (trend and seasonality), while **XGBoost** models what Prophet misses (residuals driven by external factors like promotions and holidays). The final hybrid forecast is:

```
hybrid = Prophet prediction + XGBoost residual correction
```

The core question is not "does the hybrid work" but **"is the added complexity mathematically justified?"** — answered honestly with MAE and MAPE.

## Dataset

Rossmann Store Sales — 1,017,209 daily records across 1,115 stores (Jan 2013 – Jul 2015).

| File | Description |
|------|-------------|
| `train.csv` | Daily sales per store (Sales, Customers, Open, Promo, holidays) |
| `test.csv`  | Forecast horizon (Aug 1 – Sep 17, 2015) |
| `store.csv` | Store attributes (type, assortment, competition, Promo2) |

## Approach

**1. Series Setup** — The multi-store panel is reduced to a single daily time series (total sales across open stores only). Closed days are structural zeros, not lost demand, so filtering on `Open==1` keeps the trend clean. The varying open-store count is retained as a feature.

**2. Prophet Baseline** — Fit on the aggregated series with yearly + weekly seasonality and state holidays. Establishes the number to beat.

**3. XGBoost Enhancer** — Models the Prophet residuals (`actual − predicted`) using external drivers: promo activity, holidays, calendar features, and store-mix attributes.

**4. Honest Comparison** — All three models scored on the same held-out 48-day validation window using MAE and MAPE. The split is strictly time-based (past → future); no random splitting.

## Results

| Model | MAE | MAPE |
|-------|-----|------|
| **XGBoost only** | **366,417** | **6.89%** |
| Hybrid (Prophet + XGBoost) | 425,785 | 23.98% |
| Prophet only | 487,749 | 20.75% |

## Key Finding

The two-stage hybrid **did not justify its complexity on this dataset.** A simple end-to-end XGBoost — learning trend, seasonality, and external drivers directly — outperformed both the Prophet baseline and the hybrid by a wide margin. Layering residual correction on top of Prophet actually degraded MAPE, largely because Prophet mis-estimates low-value days that inflate percentage error.

The takeaway: hybrid architectures are not universally better. When a gradient-boosted model can learn the full signal directly, the extra machinery of a two-stage pipeline adds cost without benefit. Reporting this honestly is the point of the exercise.

## Tech Stack

Python · Prophet · XGBoost · pandas · scikit-learn · matplotlib

## Bonus Explorations

- Feature-importance analysis of the residual drivers
- Lag and rolling-window features on the residual stage (leakage-safe, past values only)
- Hyperparameter tuning with `TimeSeriesSplit` cross-validation

## Usage

```bash
pip install prophet xgboost pandas scikit-learn matplotlib
jupyter notebook hybrid_forecasting.ipynb
```

Run cells top to bottom — each depends on variables defined earlier.
