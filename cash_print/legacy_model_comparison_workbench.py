# %%
#imports
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import eikon as ek
from cash_print_config import configure_eikon, data_path, today_iso
import numpy as np
import seaborn as sns

import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import optuna

# %%
#config
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')  # Updated style name for modern matplotlib versions
#configure_eikon(ek)

# %%

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
    

#Load and preprocess data
#print("Loading and preprocessing data...")
#start_date = '2022-01-07'
#end_date = today_iso()
#cash_diff, nis_aligned, margins_data, coeffs_ortho = load_and_preprocess_data(start_date, end_date)

#Feature engineering
#print("\nCreating features...")
#df_model = create_features(cash_diff, nis_aligned, margins_data, coeffs_ortho)



# %%
#pulling data 

df_model = pd.read_excel(
        data_path('df_model.xlsx'),
    )

df_model['Date'] = pd.to_datetime(df_model['Date'])

#df_model['year'] = df_model['Date'].dt.year
#df_model['month'] = df_model['Date'].dt.month
#df_model['day'] = df_model['Date'].dt.day
#df_model['weekday'] = df_model['Date'].dt.weekday
# Drop datetime column before modeling

# dates = df_model['Date']
# df_model = df_model.drop(columns=['Date'])

# # Calculate split indices
# total_len = len(df_model)
# train_size = int(total_len * 0.7)  # 75% for training
# gap_size = int(total_len * 0.1)  # 5% gap

# # Indices for train and test with a gap in between
# train_end = train_size
# test_start = train_end + gap_size

# train_df = df_model.iloc[:train_end]
# test_df = df_model.iloc[test_start:]

# train_dates = dates.iloc[:train_end]
# test_dates = dates.iloc[test_start:]

# X_train = train_df.drop(columns=['CASH_DIFF_T+1'])
# y_train = train_df['CASH_DIFF_T+1']
# X_test = test_df.drop(columns=['CASH_DIFF_T+1'])
# y_test = test_df['CASH_DIFF_T+1']
dates = df_model['Date']
df_model = df_model.drop(columns=['Date'])

# Calculate split indices for first 20% test
total_len = len(df_model)
test_size = int(total_len * 0.2)  # 20% for testing
test_end = test_size
gap_size = int(total_len * 0.1)  # 10% gap
train_start = test_end + gap_size

# Split the data
test_df = df_model.iloc[:test_end]
train_df = df_model.iloc[train_start:]

test_dates = dates.iloc[:test_end]
train_dates = dates.iloc[train_start:]

X_train = train_df.drop(columns=['CASH_DIFF_T+1'])
y_train = train_df['CASH_DIFF_T+1']
X_test = test_df.drop(columns=['CASH_DIFF_T+1'])
y_test = test_df['CASH_DIFF_T+1']


# # Train-test split (time-based, no shuffle)
# # 70/20 train/test split, discard 5% from either side of test, random test location


# total_len = len(df_model)
# test_size = int(total_len * 0.2)
# discard_size = int(test_size * 0.05)  # 5% of test size

# # Ensure test set fits within the dataset after discarding 5% on both sides
# max_start = total_len - test_size - discard_size
# min_start = discard_size
# test_start = randint(min_start, max_start)
# test_end = test_start + test_size

# # Indices for train and test
# test_indices = list(range(test_start, test_end))
# train_indices = [i for i in range(total_len) if i < test_start or i >= test_end]

# train_df = df_model.iloc[train_indices]
# test_df = df_model.iloc[test_indices]

# train_dates = dates.iloc[train_indices]
# test_dates = dates.iloc[test_indices]


# %%
#generating base models
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "XGBoost Best": XGBRegressor(
        n_estimators=241,
        max_depth=4,
        learning_rate=0.014799894470653281,
        colsample_bytree=0.9395452978984749,
        gamma=0.3552686671606645,
        reg_alpha=0.1306992146441588,
        reg_lambda=3.443295326369553,
        subsample=0.8505698792540904,
        random_state=42
    )
}

# %%
# Store predictions
comparison_df = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True)
})

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved to {model_path}")
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save predictions
    comparison_df[name] = y_pred



# %%
# Set up number of subplots based on number of models
n_models = len(models)
n_cols = 2  # Customize columns per row
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), constrained_layout=True)
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # Predict on train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Plot predictions
    ax.plot(train_dates, y_train_pred, label="Train Prediction", alpha=0.7)
    ax.plot(train_dates, y_train.values, label="Train Actual", linestyle="--", alpha=0.7)
    
    ax.plot(test_dates, y_test_pred, label="Test Prediction", alpha=0.7)
    ax.plot(test_dates, y_test.values, label="Test Actual", linestyle="--", alpha=0.7)
    
    ax.set_title(f"{name}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(True)

# If there's an unused subplot (when n_models is odd), hide it
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Model Performance on Train and Test Data", fontsize=20, y=1.02)
plt.show()


# %%
# Set up number of subplots based on number of models
n_models = len(models)
n_cols = 2  # Customize columns per row
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), constrained_layout=True)
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # Predict on train
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)

    # Plot predictions
    ax.plot(test_dates, y_test_pred, label="Train Prediction", alpha=0.7)
    ax.plot(test_dates, y_test.values, label="Train Actual", linestyle="--", alpha=0.7)

    # Annotate with R²
    ax.text(0.01, 0.95, f"R²: {r2:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    ax.set_title(f"{name}")
    ax.set_xlabel("Date" if np.issubdtype(test_dates.dtype, np.datetime64) else "Index")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(True)

# Hide unused subplots if any
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Train Set Performance for Each Model", fontsize=20, y=1.02)
plt.show()


# %%

def backtest_strategy(dates, actual, predicted, threshold=0.5, start_date=None, end_date=None, plot_title="Backtest Strategy"):
    """
    Backtests a simple threshold-based long/short strategy using predicted price movements.

    Parameters:
        actual (pd.Series): Actual price series with datetime index.
        predicted (pd.Series): Predicted price series (same index as actual).
        threshold (float): Threshold for signal generation (default=0.5).
        start_date (str or pd.Timestamp, optional): Start date for plotting/filtering.
        end_date (str or pd.Timestamp, optional): End date for plotting/filtering.
        plot_title (str): Title for the plot.

    Returns:
        dict: Dictionary with performance metrics and strategy components.
    """

    # --- Ensure input validity ---
    actual = pd.Series(actual)
    predicted = pd.Series(predicted, index=actual.index)
    actual.index = pd.to_datetime(actual.index)
    predicted.index = pd.to_datetime(predicted.index)

    # --- Filter date range if provided ---
    if start_date:
        actual = actual[actual.index >= pd.to_datetime(start_date)]
        predicted = predicted[predicted.index >= pd.to_datetime(start_date)]
    if end_date:
        actual = actual[actual.index <= pd.to_datetime(end_date)]
        predicted = predicted[predicted.index <= pd.to_datetime(end_date)]

    # --- Generate trading signals ---
    predicted_change = predicted.diff().fillna(0)
    actual_change = actual.diff().fillna(0)

    positions = np.where(predicted_change > threshold, 1,
                 np.where(predicted_change < -threshold, -1, 0))

    returns = positions[1:] * actual_change[1:]

    # --- Metrics ---
    cumulative_returns = np.cumsum(returns)
    total_return = cumulative_returns[-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else np.nan
    win_rate = np.mean(returns > 0)

    # --- Plotting (last 6 months or filtered range) ---
    df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted,
        'Position': positions
    })

    plot_range = df.copy()
    if not start_date and not end_date:
        six_months_ago = df.index.max() - pd.DateOffset(months=6)
        plot_range = df[df.index >= six_months_ago]

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Price plot with signals ---
    axs[0].plot(dates, plot_range['Actual'], label='Actual Price', color='blue', alpha=0.7)
    axs[0].plot(dates, plot_range['Predicted'], label='Predicted Price', color='orange', alpha=0.7, linestyle='--')

    # Buy/Sell arrows
    position_changes = plot_range['Position'].diff().dropna()
    for date, change in position_changes.items():
        price = plot_range.loc[date, 'Actual']
        if change == 1:
            axs[0].annotate('▲', (dates[date.value], price), color='green', fontsize=14, ha='center', va='bottom')
        elif change == -1:
            axs[0].annotate('▼', (dates[date.value], price), color='red', fontsize=14, ha='center', va='top')

    axs[0].set_ylabel("Price")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(plot_title)

    # --- Cumulative returns plot ---
    returns_series = pd.Series(returns, index=actual.index[1:])
    cum_returns_series = returns_series.cumsum()
    cum_plot = cum_returns_series[cum_returns_series.index >= plot_range.index.min()]
    axs[1].plot(dates[1:], cum_plot.values, label='Cumulative Return', color='purple')
    axs[1].set_ylabel("Cumulative Return")
    axs[1].set_xlabel("Date")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- Output metrics ---
    print(f"\nStrategy Performance:")
    print(f"Total Return: {total_return:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2%}")

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "cumulative_returns": cum_returns_series,
        "signals": pd.Series(positions, index=actual.index),
        "returns": returns_series
    }




def filter_end_of_month_low_liquidity(dates, actual, predicted, n_days=10):
    """
    Remove the last n_days of each month from the backtest to moderate for low liquidity.
    Returns filtered dates, actual, predicted.
    """
    df = pd.DataFrame({'Date': dates, 'Actual': actual, 'Predicted': predicted})
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    last_days = df.groupby(['year', 'month'])['day'].transform('max')
    mask = df['day'] <= (last_days - n_days)
    df = df[mask]
    return df['Date'], df['Actual'], df['Predicted']

# In your backtesting loop, apply the filter before calling backtest_strategy
performance_summary = []

for name, model in models.items():
    print(f"\nBacktesting model: {name}")

    # Predict on train and test sets
    train_pred = pd.Series(model.predict(X_train), index=y_train.index)
    test_pred = pd.Series(model.predict(X_test), index=y_test.index)

    # --- Filter out end-of-month low liquidity days ---
    filtered_train_dates, filtered_y_train, filtered_train_pred = filter_end_of_month_low_liquidity(
        train_dates, y_train, train_pred, n_days=3
    )
    filtered_test_dates, filtered_y_test, filtered_test_pred = filter_end_of_month_low_liquidity(
        test_dates, y_test, test_pred, n_days=3
    )

    # Backtest on train
    print(f"\n--- {name} | Train Set ---")
    train_result = backtest_strategy(
        dates=filtered_train_dates,
        actual=filtered_y_train,
        predicted=filtered_train_pred,
        threshold=0.5,
        plot_title=f"{name} - Train Set Backtest"
    )

    # Backtest on test
    print(f"\n--- {name} | Test Set ---")
    test_result = backtest_strategy(
        dates=filtered_test_dates,
        actual=filtered_y_test,
        predicted=filtered_test_pred,
        threshold=0.5,
        plot_title=f"{name} - Test Set Backtest"
    )

    # Store results in summary table
    performance_summary.append({
        "Model": name,
        "Train Return": train_result["total_return"],
        "Train Sharpe": train_result["sharpe_ratio"],
        "Train Win Rate": train_result["win_rate"],
        "Test Return": test_result["total_return"],
        "Test Sharpe": test_result["sharpe_ratio"],
        "Test Win Rate": test_result["win_rate"]
    })

# Create DataFrame of all model performance
performance_df = pd.DataFrame(performance_summary)
performance_df = performance_df.sort_values(by="Test Return", ascending=False)

performance_df = performance_df.sort_values(by="Test Return", ascending=False)



# %%
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score

# #optimize xgboost model using GridSearchCV
# xgb_base = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [5, 7, 10],
#     'gamma': [0.1, 0.2, 0.3],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'subsample': [0.3, 0.6],
#     'colsample_bytree': [0.5],
#     'reg_alpha': [0, 0.3],
#     'reg_lambda': [0, 0.3]
# }

# grid_search = GridSearchCV(
#     estimator=xgb_base,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
# # Get the best parameters and score
# best_params = grid_search.best_params_
# best_score = -grid_search.best_score_
# print(f"Best parameters: {best_params}")
# print(f"Best cross-validated MSE: {best_score:.4f}")

# #backtest the optimized XGBoost model
# print("\nBacktesting optimized XGBoost model...")
# xgb_optimized = XGBRegressor(**best_params, random_state=42)
# xgb_optimized.fit(X_train, y_train)
# train_pred = pd.Series(xgb_optimized.predict(X_train), index=y_train.index)
# test_pred = pd.Series(xgb_optimized.predict(X_test), index=y_test.index)
# train_result = backtest_strategy(
#     dates=train_dates,
#     actual=y_train,
#     predicted=train_pred,
#     threshold=0.5,
#     plot_title="Optimized XGBoost - Train Set Backtest"
# )
# test_result = backtest_strategy(
#     dates=test_dates,
#     actual=y_test,
#     predicted=test_pred,
#     threshold=0.5,
#     plot_title="Optimized XGBoost - Test Set Backtest"
# )
# # Analyze features with PCA
# def analyze_features(X, y, model, apply_pca=False, n_components=0.95):
#     """Analyze features and optionally apply PCA"""
#     # Fit the model
#     model.fit(X, y)
    
#     # Feature importance
#     feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
#     print("Feature Importances:\n", feature_importances)

#     if apply_pca:
#         # Standardize features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Apply PCA
#         pca = PCA(n_components=n_components)
#         X_reduced = pca.fit_transform(X_scaled)

#         print(f"PCA reduced features to {X_reduced.shape[1]} components.")
#         return pd.DataFrame(X_reduced, index=X.index)
    
#     return X

# analyze_features(X_train, y_train, xgb_base, apply_pca=True, n_components=0.95)
# X_test_reduced = analyze_features(X_test, y_test, xgb_base, apply_pca=True, n_components=0.95)
# # Train and evaluate models


# # %%
# import optuna
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np


# def objective(trial):
#     model = XGBRegressor(
#         n_estimators=trial.suggest_int("n_estimators", 100, 1000),
#         max_depth=trial.suggest_int("max_depth", 3, 10),
#         learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
#         colsample_bytree=trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
#         gamma=trial.suggest_uniform("gamma", 0, 1),
#         reg_alpha=trial.suggest_uniform("reg_alpha", 0, 1),
#         reg_lambda=trial.suggest_uniform("reg_lambda", 0, 10),
#         subsample=trial.suggest_uniform("subsample", 0.5, 1.0),
#         random_state=42,
#         n_jobs=-1
#     )

#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds) ** 0.5  # <-- FIXED HERE
#     return rmse


# # --- Step 3: Run the Optimization ---
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# # --- Step 4: Print Best Result ---
# print("Best params:", study.best_params)
# print("Best RMSE:", study.best_value)


