# Naphtha Cash Print Research

This folder contains a cleaned research layer for the naphtha cash print / MOC differential work.

The project goal is to test whether observable market structure features can forecast next-session naphtha cash differential changes well enough to support a simple directional trading rule.

## Research Question

Can forward-curve shape, refinery margin indicators, NIS data, and lagged cash differential behavior improve forecasts of the next naphtha cash print/MOC differential versus simple baselines?

## Current CV Framing

Developed a Python commodities research framework for naphtha cash differential forecasting using Eikon market data, forward-curve shape factors, refinery margin indicators, NIS data, lagged market features, XGBoost/ensemble models, SHAP explainability, and walk-forward backtesting.

## Clean Workflow

1. Refresh `forward_curve.xlsx` from Eikon with `pull_forward_curve_history.py` (needed before the curve-shape orthogonal factors can cover the current date range).
2. Build or update `df_model.xlsx` from raw Eikon and local desk data.
3. Run `run_cash_diff_research.py` against the prepared model dataset.
4. Review the generated model metrics, feature manifest, cost-sensitivity backtest, and prediction history.
5. Compare models against zero-change and rolling mean-reversion baselines before making any performance claim.

Refresh the model dataset on a machine with Refinitiv Eikon/Workspace running:

Option A: put your key in the gitignored local config file:

```bash
cp cash_print/local_config.example.json cash_print/local_config.json
```

Then edit `cash_print/local_config.json`:

```json
{
  "EIKON_APP_KEY": "your-eikon-app-key"
}
```

Option B: set it as an environment variable:

```bash
export EIKON_APP_KEY="your-eikon-app-key"
```

Then run:

```bash
python3 cash_print/pull_forward_curve_history.py \
  --output cash_print/forward_curve.xlsx

python3 cash_print/build_model_dataset.py \
  --output cash_print/df_model.xlsx
```

Both scripts default to `--start-date 2022-01-07` and `--end-date` equal to today's date.

`pull_forward_curve_history.py` pulls the full daily history for each of the
12 rolling monthly naphtha CIF NWE Cargo Financial contracts (Mo01, the front
month, through Mo12) via `ek.get_timeseries`, one bulk call per contract, and
writes them to the `curve` sheet as one row per business day and one column
per month-forward offset (1..12). An earlier version tried to snapshot the
`0#NAF-NWE:` chain day by day, but that chain does not support historical
point-in-time snapshots for roughly the last several months, so it silently
failed for recent dates; the rolling-contract RICs (Mo01 is the same RIC used
as `MOC` elsewhere in this project) don't have that problem. `build_model_dataset.py`
then fits QR-orthogonalized polynomial coefficients (`ortho_coef_0..3`, the
curve's orthogonal level/slope/curvature factors) to each day's curve shape
and folds them into `df_model.xlsx` along with their lags.

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

- `cash_print_config.py`: shared paths, date defaults, and Eikon config lookup.
- `local_config.example.json`: safe template for local Eikon credentials.
- `local_config.json`: your gitignored local Eikon credential file; do not commit this.
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
- `pull_forward_curve_history.py`: incremental daily Eikon forward-curve history refresh for `forward_curve.xlsx`'s `curve` sheet.

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
