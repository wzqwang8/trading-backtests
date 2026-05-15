# Naphtha Cash Print Research

This folder contains a cleaned research layer for the naphtha cash print / MOC differential work.

The project goal is to test whether observable market structure features can forecast next-session naphtha cash differential changes well enough to support a simple directional trading rule.

## Research Question

Can forward-curve shape, refinery margin indicators, NIS data, and lagged cash differential behavior improve forecasts of the next naphtha cash print/MOC differential versus simple baselines?

## Current CV Framing

Developed a Python commodities research framework for naphtha cash differential forecasting using Eikon market data, forward-curve shape factors, refinery margin indicators, NIS data, lagged market features, XGBoost/ensemble models, SHAP explainability, and walk-forward backtesting.

## Clean Workflow

1. Build or update `df_model.xlsx` from the raw Eikon and desk data.
2. Run `cash_print_research.py` against the prepared model dataset.
3. Review the generated model metrics, backtest metrics, and prediction history.
4. Compare models against persistence and rolling-mean baselines before making any performance claim.

## Expected Input

The research runner expects a tabular model file such as `df_model.xlsx` or `df_model.csv` with:

- a date column, preferably `Date`;
- a target column, preferably `CASH_DIFF_T+1`;
- feature columns known at prediction time.

The script avoids hardcoded Eikon credentials and local desk paths. Use environment variables and local config in the data-preparation layer instead.

## Outputs

By default the runner writes to `cash_print/research_outputs/`:

- `model_metrics.csv`: RMSE, MAE, R2, directional accuracy, and train/test sizes;
- `backtest_metrics.csv`: total return, Sharpe, max drawdown, turnover, and win rate;
- `prediction_history.csv`: actual, predicted, signal, position, and strategy return by date.

## Quality Bar Before CV Claims

A result is CV-ready only if:

- test-period metrics beat persistence and rolling-mean baselines;
- backtest performance survives transaction-cost sensitivity;
- validation is chronological or walk-forward, with no train/test leakage;
- signals use only information available at decision time;
- the result table includes drawdown, Sharpe, win rate, turnover, and number of trades.
