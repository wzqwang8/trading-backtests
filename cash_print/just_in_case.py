import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eikon as ek
from numpy.linalg import qr
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
import shap

# Set display option
pd.set_option('display.max_columns', None)

########################
# 1. SETTINGS & DATA LOAD
########################

# Eikon API Key
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Date range
start = '2022-01-07'
end = '2025-05-28'

# Download price data from Eikon and align indexes
prin_close = ek.get_timeseries('PAAAL00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
moc_close = ek.get_timeseries('PAAAJ00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})

prin_close.sort_index(inplace=True)
moc_close.sort_index(inplace=True)

# Calculate CASH DIFF
cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH DIFF')
print('cash diff')
print(cash_diff)

# Load NIS and prepare daily forward-filled series aligned to cash_diff
nis = pd.read_excel(
    r'M:\24.Naphtha\Python scripts\Trading back tests\Naphtha\cash_print\forward_curve.xlsx',
    sheet_name='NIS'
)
nis['date'] = pd.to_datetime(nis['date'])
nis.set_index('date', inplace=True)
nis_daily = nis.reindex(pd.date_range(nis.index.min(), nis.index.max(), freq='D')).ffill()
nis_aligned = nis_daily.reindex(cash_diff.index).ffill()
print('nis')
print(nis_aligned)
# Refinery margins
gas = ek.get_timeseries('GLBOBOXYO=ARG', start_date=start, end_date=end)['CLOSE']
nap = ek.get_timeseries('PAAAL00', start_date=start, end_date=end)['CLOSE']
fo_1pct = ek.get_timeseries('PUAAM00', start_date=start, end_date=end)['CLOSE']
dated_b = ek.get_timeseries('PCAAS00', start_date=start, end_date=end)['CLOSE']
go_p1pct = ek.get_timeseries('D-AAYWT00', start_date=start, end_date=end)['CLOSE']
jet = ek.get_timeseries('PJAAU00', start_date=start, end_date=end)['CLOSE']

VPR = 0.48 * nap / 8.9 + 0.245 * jet / 7.88 + 0.12 * go_p1pct / 7.45 + 0.14 * fo_1pct / 6.35 - dated_b
Topping = 0.24 * nap / 8.9 + 0.15 * jet / 7.88 + 0.28 * go_p1pct / 7.45 + 0.31 * fo_1pct / 6.35 - dated_b
Complex = 0.35 * gas / 8.33 + 0.1 * jet / 7.88 + 0.42 * go_p1pct / 7.45 + 0.07 * fo_1pct / 6.35 - dated_b
Bayernoil = 0.13 * nap / 8.9 + 0.21 * gas / 8.33 + 0.09 * jet / 7.88 + 0.485 * go_p1pct / 7.45 + 0.075 * fo_1pct / 6.35 - dated_b

margins_data = pd.DataFrame({
    'VPR': VPR,
    'Complex': Complex,
    'Topping': Topping,
    'Bayernoil': Bayernoil
}).sort_index()
print('margins')
print(margins_data)
##########################
# 2. ORTHOGONAL POLYNOMIAL FITTING
##########################
# Load forward curve and prepare polynomial basis
forwards = pd.read_excel(
    r'M:\24.Naphtha\Python scripts\Trading back tests\Naphtha\cash_print\forward_curve.xlsx',
    sheet_name='curve'
).set_index('Unnamed: 0')

dates = pd.to_datetime(forwards.columns)
start_date = dates[0]
month_offsets = (dates.year - start_date.year) * 12 + (dates.month - start_date.month)
x = np.array(month_offsets, dtype=float)

degree = 3  # Degree of polynomial

# Create Vandermonde matrix and QR decomposition
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
print('coefficients')
print(coeffs_ortho)
##########################
# 3. FEATURE ENGINEERING
##########################
n_lags = 5

# Step 1: Preserve the target
cash_diff_features = cash_diff.copy()

# Step 2: Combine all base features before lagging
features = pd.concat([cash_diff_features, margins_data], axis=1)

# Step 3: Create lagged features (from T-1 to T-n)
lagged_features = pd.DataFrame(index=features.index)
for col in features.columns:
    for lag in range(1, n_lags + 1):
        lagged_features[f'{col}_lag_{lag}'] = features[col].shift(lag)


# Step 5: Combine all aligned features
df_model = pd.concat([
    lagged_features,
    coeffs_ortho,
    nis_aligned  # already aligned, no shift
], axis=1)

# Step 6: Add target
df_model['CASH DIFF'] = cash_diff['CASH DIFF']

# Step 7: Drop NaNs from lagging
df_model.dropna(inplace=True)
print(df_model)

##########################
# 4. MODEL TRAINING & EXPLANATION - XGBoost + SHAP
##########################

# Prepare training data
X = df_model.drop(columns=['CASH DIFF'])       # keep as DataFrame
y = df_model['CASH DIFF']     
X_index = df_model.drop(columns=['CASH DIFF']).index  # Keep the datetime index
# Time-aware train/test split (80/20)
split_index = int(0.8 * len(df_model))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

feature_names = df_model.drop(columns=['CASH DIFF']).columns


# Fit XGBoost regressor
xgb_reg = XGBRegressor(random_state=42, n_estimators=100)
xgb_reg.fit(X_train, y_train)

# Predict and score on test set
r2_xgb = xgb_reg.score(X_test, y_test)
print(f"XGBoost R²: {r2_xgb:.4f}")

# SHAP explanation
explainer = shap.Explainer(xgb_reg)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=feature_names)

# Predict on test set
y_pred = xgb_reg.predict(X_test)

plt.figure(figsize=(10, 6))

# Plot actual with datetime index on x-axis
plt.plot(y_test.index, y_test.values, label='Actual', marker='o')

# Plot predicted with same x-axis, ensure aligned by index order
plt.plot(y_test.index, y_pred, label='Predicted', marker='x')

plt.title('XGBoost Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('CASH DIFF')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate dates for readability

plt.tight_layout()
plt.show()

# TimeSeries Cross-Validation with EarlyStopping
tscv = TimeSeriesSplit(n_splits=5)
r2_scores = []

plt.figure(figsize=(15, 10))  # For the plots of all folds

for i, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    # Extract dates for fold summary
    train_dates = X_index[train_idx]
    val_dates = X_index[val_idx]
    print(f"Fold {i}")
    print(f"  Train: {train_dates[0].date()} - {train_dates[-1].date()} ({len(train_idx)} rows)")
    print(f"  Test:  {val_dates[0].date()} - {val_dates[-1].date()} ({len(val_idx)} rows)\n")

    xgb_reg_cv = XGBRegressor(
        random_state=42,
        n_estimators=1000
    )

    xgb_reg_cv.fit(
        X_train_cv, y_train_cv,
        eval_set=[(X_val_cv, y_val_cv)],
        verbose=False
    )

    y_pred_val = xgb_reg_cv.predict(X_val_cv)
    r2 = r2_score(y_val_cv, y_pred_val)
    r2_scores.append(r2)

    # Plot predicted vs actual for this fold
    plt.subplot(3, 2, i)  # 5 folds, arrange in 3 rows x 2 cols grid
    plt.plot(y_val_cv.index, y_val_cv.values, label='Actual', marker='o')
    plt.plot(y_val_cv.index, y_pred_val, label='Predicted', marker='x')
    plt.title(f'Fold {i} - R²: {r2:.3f}')
    plt.xlabel('Date')
    plt.ylabel('CASH DIFF')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

print("CV R² scores:", r2_scores)
print("Mean CV R²:", np.mean(r2_scores))

####################################
####### FUTURE TEST ###############
####################################








# # Hyperparameter tuning with TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)
# alphas = np.logspace(-3, 3, 50)
# ridge_cv = GridSearchCV(Ridge(), {'alpha': alphas}, cv=tscv, scoring='r2', n_jobs=-1, refit=True)
# ridge_cv.fit(X_train, y_train)
# best_alpha = ridge_cv.best_params_['alpha']
# print(f"Best Ridge alpha: {best_alpha:.4f}")

# # Extract coefficients from Ridge pipeline
# ridge_coef = ridge.named_steps['ridge'].coef_

# coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': ridge_coef})
# coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)
# print(coef_df.head(10))

# # Fit models using best alpha for Ridge
# ols = make_pipeline(StandardScaler(), LinearRegression())
# ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))

# ols.fit(X_train, y_train)
# ridge.fit(X_train, y_train)

# # Evaluate R² on test set
# r2_ols = ols.score(X_test, y_test)
# r2_ridge = ridge.score(X_test, y_test)

# print(f"OLS R² (time-aware split):   {r2_ols:.4f}")
# print(f"Ridge R² (time-aware split): {r2_ridge:.4f}")

# # Predictions
# y_pred_ols = ols.predict(X_test)
# y_pred_ridge = ridge.predict(X_test)

# # Residuals
# residuals_ols = y_test - y_pred_ols
# residuals_ridge = y_test - y_pred_ridge

# # Residuals plot
# plt.figure(figsize=(14, 6))
# plt.plot(residuals_ols, label="OLS Residuals", alpha=0.7)
# plt.plot(residuals_ridge, label="Ridge Residuals", alpha=0.7)
# plt.axhline(0, color='black', linestyle='--', linewidth=1)
# plt.title("Residuals Over Time")
# plt.xlabel("Time Index (Test Set)")
# plt.ylabel("Residual (Actual - Predicted)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()


# # Histogram of residuals
# plt.figure(figsize=(12, 5))
# plt.hist(residuals_ols, bins=30, alpha=0.5, label="OLS")
# plt.hist(residuals_ridge, bins=30, alpha=0.5, label="Ridge")
# plt.title("Histogram of Residuals")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Ljung-Box test (null: residuals are independently distributed)
# ljung_results = acorr_ljungbox(residuals_ridge, lags=[10, 20, 30], return_df=True)
# print("Ljung-Box Test Results:")
# print(ljung_results)

