# Enhanced Naphtha Cash Diff Forecasting Model
# Combines robust feature engineering, model validation, and economic significance testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eikon as ek
from numpy.linalg import qr
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from xgboost import XGBRegressor
import shap
from datetime import timedelta
import warnings
from sklearn.model_selection import train_test_split

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')  # Updated style name for modern matplotlib versions
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

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
            r'M:\24.Naphtha\Python scripts\Trading back tests\Naphtha\cash_print\forward_curve.xlsx',
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
    
    # Load and process forward curves
    print("Processing forward curves...")
    forwards, coeffs_ortho = process_forward_curves(start_date, end_date)
    
    return cash_diff, nis_aligned, margins_data, coeffs_ortho

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

def process_forward_curves(start_date, end_date):
    """Process forward curves and generate orthogonal polynomial coefficients"""
    try:
        forwards = pd.read_excel(
            r'M:\24.Naphtha\Python scripts\Trading back tests\Naphtha\cash_print\forward_curve.xlsx',
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

def create_features(cash_diff, nis_aligned, margins_data, coeffs_ortho, n_lags=10):
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

    # Combine all features
    df_model = pd.concat([
        features,
        cash_diff,
        cash_diff_lags,
        lagged_features,
        cash_diff_target
    ], axis=1).dropna()
    print(df_model)
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
    plt.show()

########################
# 4. FEATURE ANALYSIS
########################

def analyze_features(X, y, model):
    """Perform feature importance and correlation analysis"""
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        plt.figure(figsize=(12, 6))
        importances.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Importances')
        plt.show()
    
    # Correlation matrix (for first 20 features to avoid clutter)
    corr_matrix = X.iloc[:, :20].corr().abs()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Matrix (First 20 Features)")
    plt.show()
    
    # Remove highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"\nHighly correlated features to consider dropping: {to_drop}")
    
    return X.drop(columns=to_drop, errors='ignore')

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
    plt.show()
    
    # Autocorrelation plot
    plot_acf(residuals, lags=10)  # Reduced lags for clarity
    plt.show()
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[5], return_df=True)  # Reduced lags
    print("\nLjung-Box test for residual autocorrelation:")
    print(lb_test)

########################
# 6. TRADING STRATEGY BACKTEST
########################

def backtest_strategy(actual, predicted, threshold=0.5):
    positions = np.zeros(len(actual))
    predicted_change = np.diff(predicted)
    positions[:-1][predicted_change > threshold] = 1
    positions[:-1][predicted_change < -threshold] = -1

    returns = positions[:-1] * np.diff(actual)
    cumulative_returns = np.cumsum(returns)

    # Performance metrics
    total_return = cumulative_returns[-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    win_rate = np.mean(returns > 0)

    # Prepare DataFrame
    df = pd.DataFrame({
        'Actual': actual,
        'Position': positions
    })

    # Restrict to last 6 months
    last_6m_start = df.index.max() - pd.DateOffset(months=6)
    df = df[df.index >= last_6m_start]

    # Monthly average horizontal lines
    monthly_avg = df['Actual'].resample('MS').mean()  # Month Start averages

    for i in range(len(monthly_avg)):
        start = monthly_avg.index[i]
        if i < len(monthly_avg) - 1:
            end = monthly_avg.index[i + 1]
        else:
            end = df.index[-1]
        plt.hlines(monthly_avg.iloc[i], start, end, colors='orange', linewidth=3, alpha=0.5)


    # Plot buy/sell arrows on position change
    position_diff = df['Position'].diff()
    for date, change in position_diff.dropna().items():
        price = df.loc[date, 'Actual']
        if change == 1:  # Enter long
            plt.annotate('▲', (date, price), color='green', fontsize=12, ha='center')
        elif change == -1:  # Exit long or enter short
            plt.annotate('▼', (date, price), color='red', fontsize=12, ha='center')

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

def get_forward_predictions(model, df_model, days_to_predict=5):
    """
    Get forward predictions for future dates using only features known at prediction time
    
    Args:
    model: Your trained ensemble model
    df_model: The full modeling DataFrame (must contain all needed features)
    days_to_predict: Number of days to predict forward
    """
    # Make a copy of the last available data point
    last_known = df_model.iloc[[-1]].copy()
    
    # Create a DataFrame for the predictions
    last_date = df_model.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    predictions = []
    
    # We'll use this to store our rolling window of predictions
    rolling_window = last_known.copy()
    
    for i in range(days_to_predict):
        # Prepare features - we need to ensure we're only using known information
        # This will depend on your specific feature engineering
        current_features = rolling_window.iloc[[-1]].copy()
        
        # Drop the target if it's present
        current_features = current_features.drop(columns=['CASH_DIFF_T+1'], errors='ignore')
        
        # Predict next value
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        # Create a new row for the next prediction
        new_row = current_features.copy()
        
        # Update time-dependent features
        # This is where you'd need to implement logic to update your features
        # For example, shifting lag features:
        # new_row['CASH_DIFF_T-1'] = current_features['CASH_DIFF_T0'].values[0]
        # new_row['CASH_DIFF_T0'] = pred
        
        # Append to our rolling window
        rolling_window = pd.concat([rolling_window, new_row])
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted_CASH_DIFF': predictions
    }).set_index('Date')
    
    return result



########################
# 7. MAIN EXECUTION
########################

def main():
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    start_date = '2022-01-07'
    end_date = '2025-06-03'
    cash_diff, nis_aligned, margins_data, coeffs_ortho = load_and_preprocess_data(start_date, end_date)
    
    # 2. Feature engineering
    print("\nCreating features...")
    df_model = create_features(cash_diff, nis_aligned, margins_data, coeffs_ortho)
    X = df_model.drop(columns=['CASH_DIFF_T+1'])
    y = df_model['CASH_DIFF_T+1']
    
    # 3. Train base model for feature analysis
    print("\nTraining base model for feature analysis...")
    xgb_base = XGBRegressor(random_state=42, n_estimators=100).fit(X, y)
    
    # 4. Feature analysis and selection
    print("\nAnalyzing features...")
    X_reduced = analyze_features(X, y, xgb_base)
    
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
        sample_idx = np.random.choice(X_reduced.shape[0], size=min(200, X_reduced.shape[0]), replace=False)
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_reduced.iloc[sample_idx])
        shap.summary_plot(shap_values, features=X_reduced.iloc[sample_idx], feature_names=X_reduced.columns.tolist())
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
    comparison_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred
})
    print("\nGenerating forward predictions...")
    forward_predictions = get_forward_predictions(ensemble_model, df_model, days_to_predict=5)
    print("\nNext 5 Days Predictions:")
    print(forward_predictions)
    
    # [Rest of your existing code]

    print(comparison_df)

if __name__ == "__main__":
    main()

    