# Naphtha Cash Print Research

This folder contains a cleaned research layer for the naphtha cash print / MOC differential work.

The project goal is to test whether observable market structure features can forecast next-session naphtha cash differential changes well enough to support a simple directional trading rule.

## Research Question

Can forward-curve shape, refinery margin indicators, NIS data, and lagged cash differential behavior improve forecasts of the next naphtha cash print/MOC differential versus simple baselines?

## Current CV Framing

Developed a Python commodities research framework for naphtha cash differential forecasting using Eikon market data, forward-curve shape factors, refinery margin indicators, NIS data, lagged market features, XGBoost/ensemble models, SHAP explainability, and walk-forward backtesting.

## Clean Workflow

1. Build or update `df_model.xlsx` from raw Eikon and local desk data.
2. Run `run_cash_diff_research.py` against the prepared model dataset.
3. Review the generated model metrics, feature manifest, cost-sensitivity backtest, and prediction history.
4. Compare models against zero-change and rolling mean-reversion baselines before making any performance claim.

Refresh the model dataset on a machine with Refinitiv Eikon/Workspace running:

```bash
export EIKON_APP_KEY="your-eikon-app-key"
python3 cash_print/build_model_dataset.py \
  --output cash_print/df_model.xlsx
```

The builder defaults to `--start-date 2022-01-07` and `--end-date` equal to today's date.

Example:

```bash
python3 cash_print/run_cash_diff_research.py \
  --input cash_print/df_model.xlsx \
  --output-dir cash_print/research_outputs_v2
```

## Expected Input

The research runner expects a tabular model file such as `df_model.xlsx` or `df_model.csv` with:

- a date column, preferably `Date`;
- a target column, preferably `CASH_DIFF_T+1`;
- feature columns known at prediction time.

The script avoids hardcoded Eikon credentials and local desk paths. Use environment variables and local config in the data-preparation layer instead.

## File Map

Core workflow:

- `cash_print_config.py`: shared paths, date defaults, and Eikon environment-variable setup.
- `build_model_dataset.py`: main dataset builder for `df_model.xlsx`; pulls Eikon data, reads local `forward_curve.xlsx`, creates lags and `CASH_DIFF_T+1`.
- `run_cash_diff_research.py`: clean leakage-aware research runner and walk-forward backtest.
- `df_model.xlsx`: current checked-in model dataset. As of this repo state, it runs from 2022-01-21 to 2025-07-22.
- `forward_curve.xlsx`: local forward-curve and NIS source workbook used by the builder.

Research outputs:

- `research_outputs/`: first-pass outputs from the earlier direct-level model.
- `research_outputs_v2/`: current v2 outputs from the direct-change model.

Legacy / exploratory notebooks preserved as scripts:

- `legacy_model_comparison_workbench.py`: older multi-model workbench. Note: its historical train/test split logic should not be used for claims without review.
- `draft_model_dataset_pipeline.py`: near-duplicate draft of the dataset/model pipeline.
- `prototype_full_feature_pipeline.py`: older full prototype retained for reference.
- `explore_cash_diff_curve_shape.py`: cash diff and forward-curve shape exploration.
- `explore_cash_diff_monthly_clusters.py`: grouped/monthly cash diff clustering exploration.
- `analyze_first_half_second_half.py`: first-half vs second-half monthly cash diff analysis.
- `classify_month_half_cash_diff.py`: classification experiment for monthly half-shape behavior.
- `estimate_cash_diff_volatility.py`: GARCH volatility exploration.
- `pull_forward_curve_history.py`: Eikon forward-curve history pull prototype.

Known data-builder caveat: the checked-in `df_model.xlsx` includes `EW`, `BRENT`, and `GASOLINE`. The refreshed builder reconstructs `BRENT` and `GASOLINE` from Eikon instruments already used in the legacy margin calculations. `EW` is intentionally not extended yet because the available formula was for a gasoline spread, not the naphtha EW series used here.

## Outputs

By default the improved runner writes to `cash_print/research_outputs_v2/`:

- `model_metrics.csv`: next-day-change RMSE, MAE, R2, move correlation, raw directional accuracy, and threshold-aligned signal directional accuracy;
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
- reports both raw nonzero prediction direction and thresholded signal direction, so model metrics line up with the backtest trigger.

## Quality Bar Before CV Claims

A result is CV-ready only if:

- test-period metrics beat persistence and rolling-mean baselines;
- backtest performance survives transaction-cost sensitivity;
- validation is chronological or walk-forward, with no train/test leakage;
- signals use only information available at decision time;
- the result table includes drawdown, Sharpe, active-day hit rate, turnover, and cost assumptions.
