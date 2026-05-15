"""Clean research runner for naphtha cash print forecasting.

This script consumes a prepared model dataset, compares baselines and ML models,
and writes auditable model/backtest outputs. It intentionally avoids hardcoded
Eikon credentials and desk-specific paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


TARGET_CANDIDATES = ("CASH_DIFF_T+1", "CASH DIFF T+1", "target", "Target")
DATE_CANDIDATES = ("Date", "date", "Trade Date", "Unnamed: 0")
CURRENT_CASH_DIFF_CANDIDATES = ("CASH_DIFF", "CASH DIFF", "cash_diff")


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    gap_rows: int = 5


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


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = {str(col): col for col in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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

    y = pd.to_numeric(df[target_col], errors="coerce")
    current = pd.to_numeric(df[current_col], errors="coerce")
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number]).copy()

    valid = y.notna() & current.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    current = current.loc[valid]

    return X, y, current


def chronological_split(
    X: pd.DataFrame, y: pd.Series, current: pd.Series, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    n = len(X)
    test_n = max(1, int(n * config.test_size))
    train_end = n - test_n - config.gap_rows
    if train_end <= 20:
        raise ValueError(
            f"Not enough rows for split. Rows={n}, test_n={test_n}, gap={config.gap_rows}"
        )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    current_train = current.iloc[:train_end]

    X_test = X.iloc[-test_n:]
    y_test = y.iloc[-test_n:]
    current_test = current.iloc[-test_n:]
    return X_train, X_test, y_train, y_test, current_train, current_test


def build_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=5,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def baseline_predictions(
    y_train: pd.Series, current_test: pd.Series
) -> dict[str, pd.Series]:
    rolling_mean = pd.Series(y_train).rolling(20, min_periods=5).mean().iloc[-1]
    if pd.isna(rolling_mean):
        rolling_mean = y_train.mean()

    return {
        "Persistence": current_test.copy(),
        "TrainRollingMean20": pd.Series(rolling_mean, index=current_test.index),
    }


def model_metrics(
    name: str, y_true: pd.Series, y_pred: pd.Series, train_rows: int
) -> dict[str, float | int | str]:
    actual_change = y_true.diff()
    predicted_change = y_pred.diff()
    directional_accuracy = (
        np.sign(actual_change.dropna()) == np.sign(predicted_change.dropna())
    ).mean()

    return {
        "model": name,
        "train_rows": train_rows,
        "test_rows": len(y_true),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": directional_accuracy,
    }


def backtest(
    y_true: pd.Series,
    y_pred: pd.Series,
    current_cash_diff: pd.Series,
    threshold: float,
    cost_per_turn: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    signal = y_pred - current_cash_diff
    position = pd.Series(0, index=y_true.index, dtype=float)
    position[signal > threshold] = 1.0
    position[signal < -threshold] = -1.0

    actual_change = y_true - current_cash_diff
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * actual_change - cost_per_turn * turnover
    cumulative_return = strategy_return.cumsum()
    drawdown = cumulative_return - cumulative_return.cummax()

    nonzero_returns = strategy_return[position != 0]
    return_std = strategy_return.std()
    sharpe = (
        strategy_return.mean() / return_std * np.sqrt(252)
        if return_std and not np.isnan(return_std)
        else np.nan
    )

    history = pd.DataFrame(
        {
            "actual_next_cash_diff": y_true,
            "predicted_next_cash_diff": y_pred,
            "current_cash_diff": current_cash_diff,
            "signal": signal,
            "position": position,
            "actual_change": actual_change,
            "turnover": turnover,
            "strategy_return": strategy_return,
            "cumulative_return": cumulative_return,
            "drawdown": drawdown,
        }
    )

    metrics = {
        "total_return": strategy_return.sum(),
        "annualized_sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "win_rate": (nonzero_returns > 0).mean() if len(nonzero_returns) else np.nan,
        "trade_days": int((position != 0).sum()),
        "turnover": turnover.sum(),
        "threshold": threshold,
        "cost_per_turn": cost_per_turn,
    }
    return metrics, history


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    X, y, current = prepare_features(df)
    split_config = SplitConfig(test_size=args.test_size, gap_rows=args.gap_rows)
    X_train, X_test, y_train, y_test, _, current_test = chronological_split(
        X, y, current, split_config
    )

    metrics_rows = []
    backtest_rows = []
    prediction_frames = []

    predictions = baseline_predictions(y_train, current_test)
    for name, model in build_models(args.random_state).items():
        model.fit(X_train, y_train)
        predictions[name] = pd.Series(model.predict(X_test), index=y_test.index)

    for name, pred in predictions.items():
        metrics_rows.append(model_metrics(name, y_test, pred, len(y_train)))
        bt_metrics, history = backtest(
            y_test,
            pred,
            current_test,
            threshold=args.threshold,
            cost_per_turn=args.cost_per_turn,
        )
        bt_metrics["model"] = name
        backtest_rows.append(bt_metrics)
        history.insert(0, "model", name)
        prediction_frames.append(history)

    model_metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    backtest_metrics_df = pd.DataFrame(backtest_rows).sort_values(
        "total_return", ascending=False
    )
    prediction_history_df = pd.concat(prediction_frames)

    model_metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    backtest_metrics_df.to_csv(output_dir / "backtest_metrics.csv", index=False)
    prediction_history_df.to_csv(output_dir / "prediction_history.csv")

    print("\nModel metrics:")
    print(model_metrics_df.to_string(index=False))
    print("\nBacktest metrics:")
    print(backtest_metrics_df.to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate naphtha cash print forecasting models."
    )
    parser.add_argument("--input", required=True, help="Path to df_model.xlsx or CSV.")
    parser.add_argument(
        "--output-dir",
        default="cash_print/research_outputs",
        help="Directory for metrics and prediction outputs.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--gap-rows", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cost-per-turn", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
