"""Leakage-aware research runner for naphtha cash print forecasting.

The first pass in this folder predicted the next cash diff level directly with a
wide set of lagged features. This runner tests the improvement suggested by that
analysis: predict the next-day change, reduce redundant lag families using only
training data, and evaluate with walk-forward refits plus cost sensitivity.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


TARGET_CANDIDATES = ("CASH_DIFF_T+1", "CASH DIFF T+1", "target", "Target")
DATE_CANDIDATES = ("Date", "date", "Trade Date", "Unnamed: 0")
CURRENT_CASH_DIFF_CANDIDATES = ("CASH_DIFF", "CASH DIFF", "cash_diff")
LAG_PATTERN = re.compile(r"^(?P<family>.+?)[ _-]?lag[ _-]?(?P<lag>\d+)$", re.I)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    gap_rows: int = 10


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = {str(col): col for col in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file type: {path.suffix}")

    df = df.copy()
    date_col = first_existing(df.columns, DATE_CANDIDATES)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index, errors="ignore")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return numeric features, next-day level, and current level.

    Models are trained on the next-day change instead of the next-day level:
    ``CASH_DIFF_T+1 - CASH_DIFF``. The current level remains available as a
    feature because it is known at prediction time.
    """

    target_col = first_existing(df.columns, TARGET_CANDIDATES)
    if not target_col:
        raise ValueError(
            "Could not find a target column. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )

    current_col = first_existing(df.columns, CURRENT_CASH_DIFF_CANDIDATES)
    if not current_col:
        raise ValueError(
            "Could not find current cash diff column. Expected one of: "
            + ", ".join(CURRENT_CASH_DIFF_CANDIDATES)
        )

    next_level = pd.to_numeric(df[target_col], errors="coerce")
    current = pd.to_numeric(df[current_col], errors="coerce")
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number]).copy()

    valid = next_level.notna() & current.notna()
    X = X.loc[valid]
    next_level = next_level.loc[valid]
    current = current.loc[valid]

    return X, next_level, current


def split_index(n_rows: int, config: SplitConfig) -> int:
    test_n = max(1, int(n_rows * config.test_size))
    test_start = n_rows - test_n
    train_end = test_start - config.gap_rows
    if train_end <= 40:
        raise ValueError(
            f"Not enough rows for split. Rows={n_rows}, test_n={test_n}, "
            f"gap={config.gap_rows}"
        )
    return test_start


def lag_number(column: str) -> int:
    match = LAG_PATTERN.match(str(column))
    return int(match.group("lag")) if match else 0


def compress_lag_families(
    X: pd.DataFrame, keep_lags: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_columns: list[str] = []
    rows: list[dict[str, object]] = []

    for column in X.columns:
        match = LAG_PATTERN.match(str(column))
        if not match:
            kept_columns.append(column)
            rows.append(
                {
                    "feature": column,
                    "family": column,
                    "lag": np.nan,
                    "kept_after_lag_compression": True,
                    "lag_compression_reason": "not_lag_feature",
                }
            )
            continue

        lag = int(match.group("lag"))
        keep = lag in keep_lags
        if keep:
            kept_columns.append(column)
        rows.append(
            {
                "feature": column,
                "family": match.group("family").strip(),
                "lag": lag,
                "kept_after_lag_compression": keep,
                "lag_compression_reason": "kept_lag" if keep else "dropped_lag",
            }
        )

    return X.loc[:, kept_columns].copy(), pd.DataFrame(rows)


def correlation_prune(
    X_train: pd.DataFrame, threshold: float
) -> tuple[list[str], pd.DataFrame]:
    """Prune redundant columns based only on the training window."""

    numeric = X_train.apply(pd.to_numeric, errors="coerce")
    missing_rate = numeric.isna().mean()
    non_constant = numeric.nunique(dropna=True) > 1
    candidates = [column for column in numeric.columns if non_constant[column]]

    def sort_key(column: str) -> tuple[float, int, str]:
        return (float(missing_rate[column]), lag_number(str(column)), str(column))

    candidates = sorted(candidates, key=sort_key)
    corr = numeric[candidates].corr().abs() if candidates else pd.DataFrame()

    selected: list[str] = []
    dropped: dict[str, str] = {}
    for column in candidates:
        duplicate_of = None
        for selected_column in selected:
            value = corr.loc[column, selected_column]
            if pd.notna(value) and value >= threshold:
                duplicate_of = selected_column
                break
        if duplicate_of is None:
            selected.append(column)
        else:
            dropped[column] = duplicate_of

    rows = []
    for column in X_train.columns:
        rows.append(
            {
                "feature": column,
                "missing_rate_train": float(missing_rate.get(column, np.nan)),
                "unique_values_train": int(numeric[column].nunique(dropna=True)),
                "kept_after_correlation_prune": column in selected,
                "correlated_with": dropped.get(column, ""),
                "correlation_threshold": threshold,
            }
        )

    return selected, pd.DataFrame(rows)


def build_models(random_state: int, cv_gap: int) -> dict[str, object]:
    models: dict[str, object] = {
        "ElasticNet": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(max_iter=20_000, random_state=random_state),
                    ),
                ]
            ),
            param_grid={
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
            },
            scoring="neg_mean_absolute_error",
            cv=TimeSeriesSplit(n_splits=4, gap=cv_gap),
            n_jobs=1,
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=10,
                        max_features=0.8,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=250,
                        max_depth=2,
                        learning_rate=0.03,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    return models


def walk_forward_predict(
    estimator: object,
    X: pd.DataFrame,
    y_change: pd.Series,
    test_start: int,
    gap_rows: int,
    refit_every: int,
) -> np.ndarray:
    preds = np.full(len(X) - test_start, np.nan)
    for block_start in range(test_start, len(X), refit_every):
        block_end = min(len(X), block_start + refit_every)
        train_end = block_start - gap_rows
        if train_end <= 40:
            raise ValueError(
                f"Insufficient training rows before block {block_start}: {train_end}"
            )

        fitted = clone(estimator)
        fitted.fit(X.iloc[:train_end], y_change.iloc[:train_end])
        preds[block_start - test_start : block_end - test_start] = fitted.predict(
            X.iloc[block_start:block_end]
        )
    return preds


def baseline_predictions(
    current: pd.Series, test_start: int, mean_window: int
) -> dict[str, np.ndarray]:
    rolling_mean = current.rolling(mean_window, min_periods=5).mean()
    return {
        "ZeroChange": np.zeros(len(current) - test_start),
        f"MeanReversion{mean_window}": (
            rolling_mean.iloc[test_start:].to_numpy()
            - current.iloc[test_start:].to_numpy()
        ),
    }


def model_metrics(
    y_true_change: pd.Series, y_pred_change: np.ndarray, model_name: str
) -> dict[str, float | str]:
    active = np.abs(y_pred_change) > 1e-12
    if active.any():
        direction_acc = float(
            (
                np.sign(y_true_change.to_numpy()[active])
                == np.sign(y_pred_change[active])
            ).mean()
        )
    else:
        direction_acc = np.nan

    return {
        "model": model_name,
        "change_rmse": float(
            np.sqrt(mean_squared_error(y_true_change, y_pred_change))
        ),
        "change_mae": float(mean_absolute_error(y_true_change, y_pred_change)),
        "change_r2": float(r2_score(y_true_change, y_pred_change)),
        "directional_accuracy_when_active": direction_acc,
        "directional_coverage": float(active.mean()),
    }


def backtest(
    current: pd.Series,
    y_true_next_level: pd.Series,
    y_pred_change: np.ndarray,
    model_name: str,
    threshold: float,
    cost_per_turn: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    realized_change = y_true_next_level.to_numpy() - current.to_numpy()
    signal = np.where(y_pred_change > threshold, 1, np.where(y_pred_change < -threshold, -1, 0))
    turns = np.abs(np.diff(np.r_[0, signal]))
    gross = signal * realized_change
    costs = turns * cost_per_turn
    strategy_return = gross - costs

    active = signal != 0
    metrics = {
        "model": model_name,
        "threshold": threshold,
        "cost_per_turn": cost_per_turn,
        "total_return": float(strategy_return.sum()),
        "avg_daily_return": float(strategy_return.mean()),
        "daily_return_std": float(strategy_return.std(ddof=0)),
        "annualized_sharpe": float(
            np.sqrt(252) * strategy_return.mean() / strategy_return.std(ddof=0)
        )
        if strategy_return.std(ddof=0) > 0
        else np.nan,
        "hit_rate_active_days": float((gross[active] > 0).mean()) if active.any() else np.nan,
        "active_day_pct": float(active.mean()),
        "turnover": float(turns.sum()),
        "max_drawdown": float(
            (np.maximum.accumulate(strategy_return.cumsum()) - strategy_return.cumsum()).max()
        ),
    }

    history = pd.DataFrame(
        {
            "date": current.index,
            "model": model_name,
            "current_cash_diff": current.to_numpy(),
            "actual_next_cash_diff": y_true_next_level.to_numpy(),
            "actual_change": realized_change,
            "predicted_change": y_pred_change,
            "predicted_next_cash_diff": current.to_numpy() + y_pred_change,
            "signal": signal,
            "gross_return": gross,
            "cost": costs,
            "strategy_return": strategy_return,
            "cumulative_return": strategy_return.cumsum(),
        }
    )
    return metrics, history


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("At least one lag must be supplied.")
    return parsed


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    X_raw, y_next_level, current = prepare_features(df)
    y_change = y_next_level - current

    config = SplitConfig(test_size=args.test_size, gap_rows=args.gap_rows)
    test_start = split_index(len(X_raw), config)
    train_end = test_start - config.gap_rows

    keep_lags = parse_int_tuple(args.keep_lags)
    X_lag, lag_manifest = compress_lag_families(X_raw, keep_lags)
    selected_features, corr_manifest = correlation_prune(
        X_lag.iloc[:train_end], args.correlation_threshold
    )
    X = X_lag.loc[:, selected_features]

    y_test_change = y_change.iloc[test_start:]
    current_test = current.iloc[test_start:]
    y_test_next_level = y_next_level.iloc[test_start:]

    predictions: dict[str, np.ndarray] = baseline_predictions(
        current, test_start, args.mean_window
    )
    models = build_models(args.random_state, args.gap_rows)
    for model_name, model in models.items():
        predictions[model_name] = walk_forward_predict(
            estimator=model,
            X=X,
            y_change=y_change,
            test_start=test_start,
            gap_rows=args.gap_rows,
            refit_every=args.refit_every,
        )

    metrics = [
        model_metrics(y_test_change, pred, model_name)
        for model_name, pred in predictions.items()
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("change_mae")
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    costs = parse_float_list(args.costs)
    if not costs:
        raise ValueError("At least one transaction cost must be supplied.")

    backtest_rows: list[dict[str, float | str]] = []
    history_frames: list[pd.DataFrame] = []
    for cost in costs:
        for model_name, pred in predictions.items():
            row, history = backtest(
                current=current_test,
                y_true_next_level=y_test_next_level,
                y_pred_change=pred,
                model_name=model_name,
                threshold=args.threshold,
                cost_per_turn=cost,
            )
            backtest_rows.append(row)
            if cost == costs[0]:
                history_frames.append(history)

    backtests_df = pd.DataFrame(backtest_rows).sort_values(
        ["cost_per_turn", "total_return"], ascending=[True, False]
    )
    backtests_df.to_csv(output_dir / "backtest_cost_sensitivity.csv", index=False)
    backtests_df.loc[backtests_df["cost_per_turn"] == costs[0]].to_csv(
        output_dir / "backtest_metrics.csv", index=False
    )
    pd.concat(history_frames, ignore_index=True).to_csv(
        output_dir / "prediction_history.csv", index=False
    )

    feature_manifest = lag_manifest.merge(corr_manifest, on="feature", how="left")
    feature_manifest.insert(0, "initial_raw_feature_count", X_raw.shape[1])
    feature_manifest.insert(1, "after_lag_compression_count", X_lag.shape[1])
    feature_manifest.insert(2, "final_feature_count", X.shape[1])
    feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)

    print(f"Loaded {len(X_raw)} rows from {input_path}")
    print(
        "Feature counts: "
        f"raw={X_raw.shape[1]}, after_lag_compression={X_lag.shape[1]}, "
        f"final={X.shape[1]}"
    )
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))
    print("\nBacktest cost sensitivity:")
    print(backtests_df.to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to df_model.xlsx or CSV.")
    parser.add_argument(
        "--output-dir",
        default="cash_print/research_outputs_v2",
        help="Directory for CSV outputs.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--gap-rows", type=int, default=10)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument(
        "--keep-lags",
        default="1,5",
        help="Comma-separated lag horizons to keep from lag feature families.",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--mean-window", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--costs",
        default="0,0.05,0.10,0.25",
        help="Comma-separated cost-per-turn values for sensitivity testing.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
