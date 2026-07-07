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
3. Review the generated model metrics, feature manifest, cost-sensitivity backtest, and prediction history.
4. Compare models against zero-change and rolling mean-reversion baselines before making any performance claim.

Example:

```bash
python3 cash_print/cash_print_research.py \
  --input cash_print/df_model.xlsx \
  --output-dir cash_print/research_outputs_v2
```

## Expected Input

The research runner expects a tabular model file such as `df_model.xlsx` or `df_model.csv` with:

- a date column, preferably `Date`;
- a target column, preferably `CASH_DIFF_T+1`;
- feature columns known at prediction time.

The script avoids hardcoded Eikon credentials and local desk paths. Use environment variables and local config in the data-preparation layer instead.

## Outputs

By default the improved runner writes to `cash_print/research_outputs_v2/`:

- `model_metrics.csv`: next-day-change RMSE, MAE, R2, directional accuracy, and directional coverage;
- `backtest_cost_sensitivity.csv`: total return, Sharpe, drawdown, turnover, active-day rate, and hit rate across transaction-cost assumptions;
- `backtest_metrics.csv`: the first configured transaction-cost slice, kept for backwards-compatible quick inspection;
- `prediction_history.csv`: actual next cash diff, predicted change, predicted next cash diff, signal, costs, and strategy return by date;
- `feature_manifest.csv`: lag compression and train-only correlation-pruning decisions for auditability.

## Current Modeling Improvements

The runner now tests a more conservative setup suggested by the first research pass:

- predicts `CASH_DIFF_T+1 - CASH_DIFF` directly, instead of making the model relearn today’s level;
- keeps selected lag horizons by default (`lag_1` and `lag_5`) to reduce repetitive lag-family noise;
- prunes highly correlated features using only the pre-test training window;
- refits models in expanding walk-forward blocks with a configurable row gap;
- compares against `ZeroChange` and rolling mean-reversion baselines;
- writes transaction-cost sensitivity instead of relying on a single cost assumption.

## Quality Bar Before CV Claims

A result is CV-ready only if:

- test-period metrics beat persistence and rolling-mean baselines;
- backtest performance survives transaction-cost sensitivity;
- validation is chronological or walk-forward, with no train/test leakage;
- signals use only information available at decision time;
- the result table includes drawdown, Sharpe, active-day hit rate, turnover, and cost assumptions.
