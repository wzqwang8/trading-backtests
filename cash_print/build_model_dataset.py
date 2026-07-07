"""Build the model-ready cash differential dataset.

This is the main data-refresh script for ``df_model.xlsx``. It pulls cash print
and market data from Eikon, combines local forward-curve/NIS inputs, creates lag
features and the next-day target, then writes the model dataset used by
``run_cash_diff_research.py``.

Eikon Desktop/Workspace must be running locally. Set ``EIKON_APP_KEY`` in your
environment instead of hardcoding credentials.
"""

import argparse
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from numpy.linalg import qr
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from datetime import timedelta
import warnings
from sklearn.model_selection import train_test_split
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError:  # optional; only needed for --run-legacy-analysis diagnostics
    sm = None
    plot_acf = None
    acorr_ljungbox = None

try:
    import eikon as ek
except ImportError:  # pragma: no cover - environment dependent
    ek = None

try:
    import seaborn as sns
except ImportError:  # optional; only needed for --run-legacy-analysis plots
    sns = None

try:
    import shap
except ImportError:  # optional; only needed for --run-legacy-analysis explainability
    shap = None

from cash_print_config import (
    DEFAULT_START_DATE,
    configure_eikon,
    data_path,
    today_iso,
)

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
if ek is not None:
    configure_eikon(ek)
plt = None

########################
# 1. DATA LOADING & PREPROCESSING
########################

def load_and_preprocess_data(start_date, end_date):
    """Load and preprocess all required data sources"""
    
    # Load price data
    print("Loading price data from Eikon...")
    prin_close = ek.get_timeseries('PAAAL00', start_date=start_date, end_date=end_date)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
    moc_close = ek.get_timeseries('PAAAJ00', start_date=start_date, end_date=end_date)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})
    
    # Calculate CASH DIFF
    cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH_DIFF')
    
    # Load NIS data
    print("Loading NIS data...")
    try:
        nis = pd.read_excel(
            data_path('forward_curve.xlsx'),
            sheet_name='NIS'
        )
        nis['date'] = pd.to_datetime(nis['date'])
        nis.set_index('date', inplace=True)
        nis_daily = nis.reindex(pd.date_range(start=start_date, end=end_date, freq='D')).ffill()
        nis_aligned = nis_daily.reindex(cash_diff.index).ffill()
    except Exception as e:
        print(f"Error loading NIS data: {e}")
        nis_aligned = pd.DataFrame(index=cash_diff.index)
    
    # Calculate refinery margins
    print("Calculating refinery margins...")
    margins_data = calculate_refinery_margins(start_date, end_date)
    market_data = calculate_market_indicators(start_date, end_date)
    
    # Load and process forward curves
    print("Processing forward curves...")
    forwards, coeffs_ortho = process_forward_curves(start_date, end_date)
    
    return cash_diff, nis_aligned, margins_data, coeffs_ortho, market_data

def calculate_refinery_margins(start_date, end_date):
    """Calculate various refinery margins"""
    try:
        gas = ek.get_timeseries('GLBOBOXYO=ARG', start_date=start_date, end_date=end_date)['CLOSE']
        nap = ek.get_timeseries('PAAAL00', start_date=start_date, end_date=end_date)['CLOSE']
        fo_1pct = ek.get_timeseries('PUAAM00', start_date=start_date, end_date=end_date)['CLOSE']
        dated_b = ek.get_timeseries('PCAAS00', start_date=start_date, end_date=end_date)['CLOSE']
        go_p1pct = ek.get_timeseries('D-AAYWT00', start_date=start_date, end_date=end_date)['CLOSE']
        jet = ek.get_timeseries('PJAAU00', start_date=start_date, end_date=end_date)['CLOSE']

        margins = {
            'VPR': 0.48 * nap / 8.9 + 0.245 * jet / 7.88 + 0.12 * go_p1pct / 7.45 + 0.14 * fo_1pct / 6.35 - dated_b,
            'Topping': 0.24 * nap / 8.9 + 0.15 * jet / 7.88 + 0.28 * go_p1pct / 7.45 + 0.31 * fo_1pct / 6.35 - dated_b,
            'Complex': 0.35 * gas / 8.33 + 0.1 * jet / 7.88 + 0.42 * go_p1pct / 7.45 + 0.07 * fo_1pct / 6.35 - dated_b,
            'Bayernoil': 0.13 * nap / 8.9 + 0.21 * gas / 8.33 + 0.09 * jet / 7.88 + 0.485 * go_p1pct / 7.45 + 0.075 * fo_1pct / 6.35 - dated_b
        }
        
        return pd.DataFrame(margins).sort_index()
    except Exception as e:
        print(f"Error calculating margins: {e}")
        return pd.DataFrame()


def calculate_market_indicators(start_date, end_date):
    """Pull broad market features used by the checked-in research dataset.

    ``BRENT`` and ``GASOLINE`` are reconstructed from the Eikon instruments that
    were already used in the legacy margin calculations. ``EW`` is not
    reconstructed here because the original source ticker is not documented in
    the repository; if an existing df_model.xlsx has EW, main() preserves the
    historical EW values during refresh.
    """

    try:
        gasoline = ek.get_timeseries(
            'GLBOBOXYO=ARG', start_date=start_date, end_date=end_date
        )['CLOSE'].rename('GASOLINE')
        brent = ek.get_timeseries(
            'PCAAS00', start_date=start_date, end_date=end_date
        )['CLOSE'].rename('BRENT')
        return pd.concat([brent, gasoline], axis=1).sort_index()
    except Exception as e:
        print(f"Error calculating market indicators: {e}")
        return pd.DataFrame()


def process_forward_curves(start_date, end_date):
    """Process forward curves and generate orthogonal polynomial coefficients"""
    try:
        forwards = pd.read_excel(
            data_path('forward_curve.xlsx'),
            sheet_name='curve'
        ).set_index('Unnamed: 0')

        forwards.index = pd.to_datetime(forwards.index)
        forwards = forwards.loc[(forwards.index >= pd.to_datetime(start_date)) & 
                              (forwards.index <= pd.to_datetime(end_date))]

        dates = pd.to_datetime(forwards.columns)
        start_date = dates[0]
        month_offsets = (dates.year - start_date.year) * 12 + (dates.month - start_date.month)
        x = np.array(month_offsets, dtype=float)

        degree = 3  # Degree of polynomial
        V = np.vander(x, N=degree + 1, increasing=True)
        Q, R = qr(V)

        def fit_orthopoly(row):
            y = row.values
            valid = ~np.isnan(y)
            if valid.sum() < degree + 1:
                return pd.Series([np.nan] * (degree + 1))
            X = Q[valid, :]
            y_valid = y[valid]
            coeffs, _, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
            return pd.Series(coeffs, index=[f'ortho_coef_{i}' for i in range(degree + 1)])

        coeffs_ortho = forwards.apply(fit_orthopoly, axis=1)
        return forwards, coeffs_ortho
    except Exception as e:
        print(f"Error processing forward curves: {e}")
        return pd.DataFrame(), pd.DataFrame()

########################
# 2. FEATURE ENGINEERING
########################

def create_features(
    cash_diff,
    nis_aligned,
    margins_data,
    coeffs_ortho,
    market_data=None,
    n_lags=10,
    output_path=None,
):
    """Create features with lagged values and target variable"""
    
    # Target variable (next day's CASH DIFF)
    cash_diff_target = cash_diff.shift(-1).rename(columns={'CASH_DIFF': 'CASH_DIFF_T+1'})

    # Lagged features for CASH DIFF
    cash_diff_lags = pd.DataFrame(index=cash_diff.index)
    for lag in range(1, n_lags + 1):
        cash_diff_lags[f'CASH_DIFF_lag_{lag}'] = cash_diff['CASH_DIFF'].shift(lag)

    # Combine features
    features = pd.concat([coeffs_ortho, margins_data, nis_aligned], axis=1)

    # Create lagged versions of other features
    lagged_features = pd.DataFrame(index=features.index)
    for col in features.columns:
        for lag in range(1, n_lags + 1):
            lagged_features[f'{col}_lag_{lag}'] = features[col].shift(lag)

    if market_data is None:
        market_data = pd.DataFrame(index=cash_diff.index)

    # Combine all features
    df_model = pd.concat([
        features,
        cash_diff,
        cash_diff_lags,
        lagged_features,
        cash_diff_target,
        market_data,
    ], axis=1).dropna()
    print(df_model)

    output_path = output_path or data_path('df_model.xlsx')
    df_model.to_excel(output_path, index_label='Date')
    print(f"\ndf_model saved to Excel at:\n{output_path}")
    return df_model

########################
# 3. MODEL TRAINING & EVALUATION
########################

def train_and_evaluate(X, y):
    """Train and evaluate models with robust validation"""
    
    # Time-series cross-validation with gap
    tscv = TimeSeriesSplit(n_splits=5, test_size=20, gap=5)
    
    # Initialize models
    base_models = {
        'xgb': XGBRegressor(random_state=42, n_estimators=1000),
        'elastic': ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42),
        'svr': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    # Hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    print("Performing randomized search for XGBoost...")
    xgb_search = RandomizedSearchCV(
        estimator=base_models['xgb'],
        param_distributions=param_grid,
        n_iter=20,  # Reduced for faster execution
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X, y)
    best_xgb = xgb_search.best_estimator_
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Create ensemble
    print("\nTraining ensemble model...")
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', best_xgb),
            ('elastic', base_models['elastic']),
            ('svr', base_models['svr'])
        ],
        weights=[0.5, 0.3, 0.2]
    )
    ensemble.fit(X, y)
    joblib.dump(best_xgb, 'xgb_model.pkl')
    return ensemble, best_xgb

def evaluate_model(model, X, y, model_name=""):
    """Evaluate model performance with time-series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=3, test_size=20, gap=5)  # Reduced splits for faster execution
    r2_scores = []
    results = []
    
    print(f"\nEvaluating {model_name} with time-series cross-validation:")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        fold_results = pd.DataFrame({
            'date': X_test.index,
            'actual': y_test.values,
            'predicted': y_pred
        })
        results.append(fold_results)
        
        print(f"Fold {i}: R² = {r2:.3f}")
    
    print(f"\nMean R²: {np.mean(r2_scores):.3f}")
    return pd.concat(results).set_index('date'), r2_scores

def plot_predictions(results_df, title=""):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['actual'], label='Actual', alpha=0.7)
    plt.plot(results_df.index, results_df['predicted'], label='Predicted', alpha=0.7)
    plt.title(f'{title} - Actual vs Predicted CASH DIFF')
    plt.xlabel('Date')
    plt.ylabel('CASH DIFF (Next Day)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
 

########################
# 4. FEATURE ANALYSIS
########################

def analyze_features(X, y, model, n_components=None, apply_pca=False):
    """
    Perform feature importance, correlation analysis, and optional PCA.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series or np.array): Target variable
        model: Trained model with feature_importances_ attribute (if available)
        n_components (int or float): Number of PCA components or variance ratio (e.g., 0.95)
        apply_pca (bool): Whether to return PCA-transformed data

    Returns:
        pd.DataFrame or np.array: Cleaned feature set (with or without PCA)
    """
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        plt.figure(figsize=(12, 6))
        importances.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

    # Correlation matrix (for first 20 features to avoid clutter)
    corr_matrix = X.iloc[:, :20].corr().abs()
    plt.figure(figsize=(12, 10))
    if sns is not None:
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    else:
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Correlation')
    plt.title("Feature Correlation Matrix (First 20 Features)")
    plt.tight_layout()
    plt.show()

    # Remove highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"\nHighly correlated features to consider dropping: {to_drop}")
    
    X_cleaned = X.drop(columns=to_drop, errors='ignore')

    # # PCA (optional)
    # if apply_pca:
    #     print("\nApplying PCA...")
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X_cleaned)
    #     pca = PCA(n_components=n_components)
    #     X_pca = pca.fit_transform(X_scaled)

    #     explained = pca.explained_variance_ratio_
    #     cumulative = np.cumsum(explained)

    #     plt.figure(figsize=(10, 5))
    #     plt.plot(range(1, len(explained) + 1), cumulative, marker='o')
    #     plt.xlabel('Number of Principal Components')
    #     plt.ylabel('Cumulative Explained Variance')
    #     plt.title('Explained Variance by PCA Components')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    #     print(f"\nPCA reduced data to {X_pca.shape[1]} dimensions.")
    #     return X_pca  # returns numpy array

    return X_cleaned  # returns DataFrame

########################
# 5. RESIDUAL ANALYSIS
########################

def analyze_residuals(y_true, y_pred):
    """Analyze model residuals"""
    residuals = y_true - y_pred
    
    # Residual plot
    plt.figure(figsize=(12,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
 
    
    # Autocorrelation plot
    plot_acf(residuals, lags=10)  # Reduced lags for clarity
    
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[5], return_df=True)  # Reduced lags
    print("\nLjung-Box test for residual autocorrelation:")
    print(lb_test)

########################
# 6. TRADING STRATEGY BACKTEST
########################

def backtest_strategy(actual, predicted, threshold=0.5):
    # Ensure actual and predicted are pd.Series with datetime index
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)
    if not isinstance(predicted, pd.Series):
        predicted = pd.Series(predicted, index=actual.index)

    positions = np.zeros(len(actual))
    predicted_change = predicted.diff().fillna(0)  # predicted[t] - predicted[t-1]

    # Generate positions: buy (1) if predicted increase > threshold, sell (-1) if decrease > threshold
    positions[predicted_change > threshold] = 1
    positions[predicted_change < -threshold] = -1

    # Calculate returns based on position held * actual price change
    actual_change = actual.diff().fillna(0)
    returns = positions[:-1] * actual_change[1:]  # Position on day t affects return from t to t+1

    cumulative_returns = np.cumsum(returns)

    # Performance metrics
    total_return = cumulative_returns[-1]
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else np.nan
    win_rate = np.mean(returns > 0)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'Actual': actual,
        'Position': positions
    })

    # Restrict to last 6 months
    last_6m_start = df.index.max() - pd.DateOffset(months=6)
    df_6m = df[df.index >= last_6m_start]

    # Plot actual price
    plt.figure(figsize=(14,7))
    plt.plot(df_6m.index, df_6m['Actual'], label='Actual Price', color='blue')

    # Plot monthly average horizontal lines
    monthly_avg = df_6m['Actual'].resample('MS').mean()
    for i in range(len(monthly_avg)):
        start = monthly_avg.index[i]
        if i < len(monthly_avg) - 1:
            end = monthly_avg.index[i + 1]
        else:
            end = df_6m.index[-1]
        plt.hlines(monthly_avg.iloc[i], start, end, colors='orange', linewidth=2, alpha=0.6,
                   label='Monthly Avg' if i==0 else None)  # label once for legend

    # Plot buy/sell arrows where position changes occur
    position_diff = df_6m['Position'].diff()
    for date, change in position_diff.dropna().items():
        price = df_6m.loc[date, 'Actual']
        if change == 1:
            plt.annotate('▲', (date, price), color='green', fontsize=14, ha='center', va='bottom')
        elif change == -1:
            plt.annotate('▼', (date, price), color='red', fontsize=14, ha='center', va='top')

    plt.title("Trading Strategy Signals (Last 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nStrategy Performance (Full Period):")
    print(f"Total Return: {total_return:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2%}")

########################
# 7. FORWARD PREDICTIONS
########################

def build_features(df):
    """Rebuild features from scratch (e.g. lags) after appending new prediction."""
    df_feat = df.copy()
    df_feat['lag1'] = df_feat['CASH_DIFF'].shift(1)
    df_feat['lag2'] = df_feat['CASH_DIFF'].shift(2)
    # Add any other engineered features consistently here
    
    return df_feat.dropna()

def get_forward_predictions(model, cutoff_df, days_to_predict=5):
    forward_preds = []
    current_df = cutoff_df.copy()

    for _ in range(days_to_predict):
        # Rebuild features using the latest version of current_df
        current_features = build_features(current_df)

        # Match only the features used in training
        X_input = current_features[model.feature_names_in_]

        # Predict using the last available row
        pred = model.predict(X_input.iloc[[-1]])[0]
        forward_preds.append(pred)

        # Append predicted value to current_df
        next_date = current_df.index[-1] + pd.Timedelta(days=1)
        next_row = pd.Series({'CASH_DIFF': pred}, name=next_date)
        current_df = pd.concat([current_df, next_row.to_frame().T])

    return forward_preds



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=today_iso())
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(data_path("df_model.xlsx")),
        help="Where to write the refreshed model dataset.",
    )
    parser.add_argument(
        "--run-legacy-analysis",
        action="store_true",
        help="Also run the older in-script model diagnostics after building df_model.",
    )
    return parser.parse_args()


def preserve_existing_overlays(output_path):
    """Preserve checked-in columns whose data source is not yet documented."""

    try:
        existing = pd.read_excel(output_path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Could not read existing overlays from {output_path}: {e}")
        return pd.DataFrame()

    if "Date" not in existing.columns:
        return pd.DataFrame()

    overlay_columns = [col for col in ["EW"] if col in existing.columns]
    if not overlay_columns:
        return pd.DataFrame()

    overlays = existing[["Date", *overlay_columns]].copy()
    overlays["Date"] = pd.to_datetime(overlays["Date"], errors="coerce")
    overlays = overlays.dropna(subset=["Date"]).set_index("Date")
    return overlays


########################
# 7. MAIN EXECUTION
########################

def main():
    args = parse_args()

    if ek is None:
        raise SystemExit(
            "Cannot refresh df_model.xlsx because the 'eikon' Python package is "
            "not installed in this environment. Run this script on a machine with "
            "Refinitiv Eikon/Workspace and the eikon package installed."
        )

    if args.run_legacy_analysis:
        global plt
        import matplotlib.pyplot as plt

        plt.style.use('seaborn-v0_8')

    output_path = data_path(args.output) if not Path(args.output).is_absolute() else Path(args.output)

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    cash_diff, nis_aligned, margins_data, coeffs_ortho, market_data = load_and_preprocess_data(
        args.start_date, args.end_date
    )

    overlays = preserve_existing_overlays(output_path)
    if not overlays.empty:
        market_data = market_data.join(overlays, how="left")
        print(
            "Preserved existing overlay columns without documented source: "
            + ", ".join(overlays.columns)
        )

    # 2. Feature engineering
    print("\nCreating features...")
    df_model = create_features(
        cash_diff,
        nis_aligned,
        margins_data,
        coeffs_ortho,
        market_data=market_data,
        n_lags=args.lags,
        output_path=output_path,
    )
    if not args.run_legacy_analysis:
        print("\nDataset build complete. Use run_cash_diff_research.py for model testing.")
        return

    X = df_model.drop(columns=['CASH_DIFF_T+1'])
    y = df_model['CASH_DIFF_T+1']

    # 3. Train base model for feature analysis
    print("\nTraining base model for feature analysis...")
    xgb_base = XGBRegressor(random_state=42, n_estimators=100).fit(X, y)

    # 4. Feature analysis and selection, with PCA
    print("\nAnalyzing features...")
    X_reduced = analyze_features(X, y, xgb_base, apply_pca=True, n_components=0.95)
    # X_reduced = pd.DataFrame(X_reduced, index=X.index, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])

    # 5. Train and evaluate models
    print("\nTraining models...")
    ensemble_model, xgb_model = train_and_evaluate(X_reduced, y)
    
    # Evaluate ensemble
    ensemble_results, _ = evaluate_model(ensemble_model, X_reduced, y, "Ensemble Model")
    plot_predictions(ensemble_results, "Ensemble Model")
    
    # Evaluate XGBoost
    xgb_results, _ = evaluate_model(xgb_model, X_reduced, y, "XGBoost Model")
    plot_predictions(xgb_results, "XGBoost Model")
    
    # 6. Residual analysis
    print("\nAnalyzing residuals...")
    y_pred = ensemble_model.predict(X_reduced)
    analyze_residuals(y, y_pred)
    
    # 7. Strategy backtest
    print("\nRunning strategy backtest...")
    backtest_strategy(y, y_pred)
    
    # 8. SHAP analysis (on smaller sample for performance)
    print("\nRunning SHAP analysis (on sample)...")
    try:
        if shap is None:
            raise ImportError("shap is not installed")
        sample_idx = np.random.choice(X_reduced.shape[0], size=min(200, X_reduced.shape[0]), replace=False)
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_reduced.iloc[sample_idx])
        shap.summary_plot(shap_values, features=X_reduced.iloc[sample_idx], feature_names=X_reduced.columns.tolist())
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
    
    print("\nGenerating forward predictions...")
    cutoff_df = df_model.iloc[:-5]

    # Run forward predictions
    comparison_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred
})
    print(comparison_df)
    forward_preds = get_forward_predictions(xgb_model, cutoff_df, days_to_predict=5)
    print("\nNext 5 Days Predictions:")
    print(forward_preds)
    


if __name__ == "__main__":
    main()

