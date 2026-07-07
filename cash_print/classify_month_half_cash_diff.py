import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eikon as ek
from cash_print_config import configure_eikon, data_path, today_iso
from numpy.linalg import qr
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap

# Set display option
pd.set_option('display.max_columns', None)

########################
# 1. SETTINGS & DATA LOAD
########################

# Eikon API Key
configure_eikon(ek)

# Date range
start = '2022-01-07'
end = today_iso()

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
    data_path('forward_curve.xlsx'),
    sheet_name='NIS'
)
nis['date'] = pd.to_datetime(nis['date'])
nis.set_index('date', inplace=True)
nis_daily = nis.reindex(pd.date_range(start=start, end=end, freq='D')).ffill()
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
    data_path('forward_curve.xlsx'),
    sheet_name='curve'
).set_index('Unnamed: 0')

forwards.index = pd.to_datetime(forwards.index)
forwards = forwards.loc[(forwards.index >= pd.to_datetime(start)) & 
                        (forwards.index <= pd.to_datetime(end))]

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
# 3. FEATURE ENGINEERING (MODIFIED FOR FORECASTING)
##########################
n_lags = 5

# Step 1: Create lagged features for CASH_DIFF
cash_diff_lags = pd.DataFrame(index=cash_diff.index)
for lag in range(1, n_lags + 1):
    cash_diff_lags[f'CASH_DIFF_lag_{lag}'] = cash_diff['CASH DIFF'].shift(lag)

# Step 2: Create lagged versions of other features
lagged_features = pd.DataFrame(index=coeffs_ortho.index)
for col in coeffs_ortho.columns:
    for lag in range(1, n_lags + 1):
        lagged_features[f'{col}_lag_{lag}'] = coeffs_ortho[col].shift(lag)

# Step 3: Combine all features
df_all = pd.concat([
    cash_diff_lags,
    coeffs_ortho,
    margins_data,
    lagged_features,
    nis_aligned
], axis=1)

# Step 4: Add target - compute monthly H1/H2 split
df_all['CASH_DIFF'] = cash_diff['CASH DIFF']
df_all['month'] = df_all.index.to_period('M')
df_all['day'] = df_all.index.day
df_all['half'] = np.where(df_all['day'] <= 15, 'H1', 'H2')

# Step 5: Compute H1 and H2 mean for each month
monthly_diff = df_all.groupby(['month', 'half'])['CASH_DIFF'].mean().unstack()

# Step 6: Create binary target: H2 > H1
monthly_diff['target'] = (monthly_diff['H2'] > monthly_diff['H1']).astype(int)
print(monthly_diff)
# Step 7: For each month, select features as of 15th of the month (or earlier)
def get_snapshot_features(df, target_month):
    # Get data up to the 15th of the month
    snapshot = df[(df.index.to_period('M') == target_month) & (df.index.day <= 15)]
    if snapshot.empty:
        return None
    return snapshot.mean(numeric_only=True)

snapshots = []
for month in monthly_diff.index:
    snapshot = get_snapshot_features(df_all, month)
    if snapshot is not None:
        snapshot['month'] = str(month)
        snapshots.append(snapshot)

# Step 8: Combine all snapshots
df_features = pd.DataFrame(snapshots).set_index('month')
print(df_features)

# Step 9: Merge with targets
df_features.index = pd.PeriodIndex(df_features.index, freq='M')
df_final = df_features.merge(monthly_diff[['target']], left_index=True, right_index=True)


##########################
# 4. MODEL TRAINING & EXPLANATION - XGBoost + SHAP
##########################

# Step 10: Shift target to predict next month's behavior
df_final['target_next_month'] = df_final['target'].shift(-1)
df_final.dropna(subset=['target_next_month'], inplace=True)

# Step 11: Set features and labels
feature_cols = df_final.drop(columns=['target', 'target_next_month']).columns.tolist()
X_all = df_final[feature_cols].values
y_all = df_final['target_next_month'].values

# Step 12: Train/test split
split_index = int(len(df_final) * 0.8)
X_train, X_test = X_all[:split_index], X_all[split_index:]
y_train, y_test = y_all[:split_index], y_all[split_index:]

# Step 13: Train model
# Calculate scale_pos_weight = #negative / #positive
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Step 14: Evaluate
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]
print("Classification Report (Simple Split):")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Step 15: SHAP explanation
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, features=X_test, feature_names=feature_cols)

shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_imp_df = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': shap_importance
}).sort_values(by='shap_importance', ascending=False)
print("\nTop SHAP features:")
print(shap_imp_df.head(20))

##########################
# 5. WALK FORWARD CV with Hyperparameter Tuning
##########################

n_obs = len(df_final)
oof_preds = np.zeros(n_obs)
oof_proba = np.zeros(n_obs)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}

min_train_size = 12

for i in range(min_train_size, n_obs):
    X_train_fold, y_train_fold = X_all[:i], y_all[:i]
    X_val_fold, y_val_fold = X_all[i:i+1], y_all[i:i+1]

    model = XGBClassifier(eval_metric='logloss', random_state=42)

    rand_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='roc_auc',
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    rand_search.fit(X_train_fold, y_train_fold)
    best_model = rand_search.best_estimator_

    oof_preds[i] = best_model.predict(X_val_fold)[0]
    oof_proba[i] = best_model.predict_proba(X_val_fold)[0, 1]

# Step 16: Evaluate walk-forward
valid_mask = np.arange(n_obs) >= min_train_size
print("\nWalk-Forward CV Results:")
print("Classification Report:")
print(classification_report(y_all[valid_mask], oof_preds[valid_mask]))
print("ROC AUC:", roc_auc_score(y_all[valid_mask], oof_proba[valid_mask]))

# Step 17: Fit final model for latest prediction
X_train_final = X_all[:n_obs - 1]
y_train_final = y_all[:n_obs - 1]
X_test_final = X_all[n_obs - 1:]

final_model = XGBClassifier(eval_metric='logloss', random_state=42)
final_model.set_params(**rand_search.best_params_)
final_model.fit(X_train_final, y_train_final)

# Step 18: SHAP on final prediction
explainer = shap.Explainer(final_model)
shap_values = explainer(X_test_final)
shap.summary_plot(shap_values, features=X_test_final, feature_names=feature_cols)

##########################
# 6. PREDICTION FOR JUNE 2025
##########################

last_available = '2025-05_H1'

if last_available not in df_final.index:
    raise ValueError(f"{last_available} not found in df_final. Available half-months: {df_final.index[-5:].tolist()}")

latest_features = df_final.loc[[last_available], feature_cols]
june_pred_class = final_model.predict(latest_features)[0]
june_pred_proba = final_model.predict_proba(latest_features)[0, 1]

print("\n\U0001F52E Prediction for June 2025:")
print(f"\U0001F9FE Class Prediction: {'H1 > H2' if june_pred_class == 1 else 'H1 <= H2'}")
print(f"\U0001F4CA Probability that H1 > H2: {june_pred_proba:.2%}")
