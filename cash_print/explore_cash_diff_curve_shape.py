import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import eikon as ek
from cash_print_config import configure_eikon, data_path, today_iso
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from datetime import timedelta
pd.set_option('display.max_columns', None)

# Eikon API Key
configure_eikon(ek)

#################################################
start = '2024-01-01'
end = today_iso()

# Month mappings
prin_close = ek.get_timeseries('PAAAL00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
moc_close = ek.get_timeseries('PAAAJ00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})


# Calculate CASH DIFF
cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH DIFF')

# Plot the CASH DIFF time series
plt.figure(figsize=(14, 6))
plt.plot(cash_diff.index, cash_diff['CASH DIFF'], label='CASH DIFF', color='blue', linewidth=2)

plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('CASH DIFF: PRINT vs MOC', fontsize=14)
plt.xlabel('Date')
plt.ylabel('CASH DIFF ($/mt)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Load the Excel file
forwards = pd.read_excel(
    data_path('forward_curve.xlsx'),
    sheet_name='curve'
)
forwards = forwards.set_index('Unnamed: 0')

# X-values: month offsets
x_vals = np.arange(forwards.shape[1])

# Fit polynomial per row, skipping NaNs
def fit_polynomial_to_row(row, degree=3):
    y = row.values.astype(float)
    mask = ~np.isnan(y)
    x_valid = x_vals[mask]
    y_valid = y[mask]
    if len(y_valid) > degree:  # Need enough points to fit
        return np.polyfit(x_valid, y_valid, degree)
    else:
        return [np.nan] * (degree + 1)

# Apply per row
coeff_df = forwards.apply(lambda row: fit_polynomial_to_row(row, degree=3), axis=1)
coeff_df = pd.DataFrame(coeff_df.tolist(), index=forwards.index,
                        columns=[f'coef_d{i}' for i in range(3, -1, -1)])

print(coeff_df)





# Combine into one DataFrame
df = pd.concat([prin_close, moc_close, cash_diff, coeff_df], axis=1)

# Drop rows with all NaNs (optional)
df.dropna(how='all', inplace=True)
print(df)
# # Define time ranges
# end_date = pd.to_datetime("today").normalize()
# date_ranges = [
#     (end_date - pd.DateOffset(years=1), end_date),
#     (end_date - pd.DateOffset(years=2), end_date),
#     (end_date - pd.DateOffset(years=5), end_date)
# ]

# titles = ["Past year", "Past 2 years", "Past 5 years"]

# # Set up the subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# for i, ((start_range, end_range), title) in enumerate(zip(date_ranges, titles)):
#     # Filter data for the period
#     df_period = df[(df.index >= start_range) & (df.index <= end_range)].copy()

#     # Reset and prepare
#     df_reset = df_period.reset_index()
#     df_reset['Month'] = df_reset['Date'].dt.to_period('M')
#     df_valid = df_reset.dropna(subset=['CASH DIFF']).copy()
#     df_valid['Pricing Day'] = df_valid.groupby('Month').cumcount() + 1

#     # Calculate derivative and abs derivative
#     df_valid['Derivative'] = df_valid.groupby('Month')['CASH DIFF'].diff()
#     df_valid = df_valid.dropna(subset=['Derivative'])
#     df_valid['Abs Derivative'] = df_valid['Derivative'].abs()

#     # Group by pricing day and calculate mean
#     # Group by pricing day and calculate means
#     avg_cash_diff = df_valid.groupby('Pricing Day')['CASH DIFF'].mean()
#     avg_abs_derivative = df_valid.groupby('Pricing Day')['Abs Derivative'].mean()
#     max_day = avg_abs_derivative.idxmax()
#     max_value = avg_abs_derivative.max()

#     # Plot both lines
#     axs[i].plot(avg_abs_derivative.index, avg_abs_derivative.values, marker='o', linestyle='-', color='purple', label='Avg |Δ CASH DIFF|')
#     axs[i].plot(avg_cash_diff.index, avg_cash_diff.values, marker='x', linestyle='--', color='blue', label='Avg CASH DIFF')

#     # Mark the max of abs derivative
#     axs[i].axvline(max_day, color='red', linestyle=':', linewidth=1)
#     axs[i].annotate(f'Day {max_day}\n{max_value:.2f}',
#                     xy=(max_day, max_value),
#                     xytext=(max_day + 1, max_value + 0.5),
#                     arrowprops=dict(facecolor='red', arrowstyle='->'),
#                     fontsize=9, color='red')

#     axs[i].set_title(title)
#     axs[i].set_xlabel('Pricing Day')
#     axs[i].set_xticks(range(1, df_valid['Pricing Day'].max() + 1))
#     axs[i].grid(True)
#     axs[i].legend()

# axs[0].set_ylabel('Average |Δ Cash Diff| ($/mt)')
# plt.suptitle('Average |Δ Cash Diff| by Pricing Day\nAcross Diff Windows', fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])



# #####################################################################
# df_monthly_shapes = df_valid.pivot(index='Month', columns='Pricing Day', values='CASH DIFF')
# df_monthly_shapes_filled = df_monthly_shapes.fillna(0)

# # --- Standardize ---
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df_monthly_shapes_filled)

# # --- Find best number of clusters using silhouette score ---
# sil_scores = []
# k_range = range(2, 10)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     labels = kmeans.fit_predict(df_scaled)
#     score = silhouette_score(df_scaled, labels)
#     sil_scores.append(score)

# # Plot silhouette scores
# plt.figure(figsize=(6, 4))
# plt.plot(k_range, sil_scores, marker='o')
# plt.title("Silhouette Score vs Number of Clusters")
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette Score")
# plt.grid(True)
# plt.tight_layout()


# # --- Choose best k ---
# best_k = k_range[sil_scores.index(max(sil_scores))]
# print(f"Best number of clusters based on silhouette score: {best_k}")

# # --- Run final KMeans with best_k ---
# kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
# cluster_labels = kmeans.fit_predict(df_scaled)

# # Add cluster back to data
# df_monthly_shapes_filled['Cluster'] = cluster_labels
# df_valid['Cluster'] = df_valid['Month'].map(df_monthly_shapes_filled['Cluster'])

# # --- Plot average shape per cluster ---
# pricing_day_columns = [col for col in df_monthly_shapes.columns if col not in ['Cluster']]

# plt.figure(figsize=(10, 6))
# for cluster in range(best_k):
#     cluster_rows = df_monthly_shapes_filled[df_monthly_shapes_filled['Cluster'] == cluster]
#     cluster_mean = cluster_rows[pricing_day_columns].mean()
#     plt.plot(pricing_day_columns, cluster_mean.values, label=f'Cluster {cluster}', marker='o')

# plt.title('Average CASH DIFF Shape by Cluster')
# plt.xlabel('Pricing Day')
# plt.ylabel('CASH DIFF (USD/MT)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()


# # --- Print months in each cluster ---
# cluster_groups = df_valid.groupby('Cluster')['Month'].unique()
# for cluster, months in cluster_groups.items():
#     print(f"Cluster {cluster}: {list(months)}")


# ###############################################
# ####### PLOT MONTHLY ################
# # Prepare the data
# df_reset = df.reset_index()
# df_reset['Month'] = df_reset['Date'].dt.to_period('M')

# # Only consider days where CASH DIFF is not NaN
# df_valid = df_reset.dropna(subset=['CASH DIFF']).copy()

# # Assign pricing day per month (1, 2, 3...) only on valid pricing days
# df_valid['Pricing Day'] = df_valid.groupby('Month').cumcount() + 1

# # Now continue with the plotting as before
# months = sorted(df_valid['Month'].unique())
# n_months = len(months)

# cols = 5
# rows = int(np.ceil(n_months / cols))
# fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=False, sharey=True)
# axes = axes.flatten()

# for i, month in enumerate(months):
#     ax = axes[i]
#     month_df = df_valid[df_valid['Month'] == month]
#     # Compute derivative of CASH DIFF
#     month_df['Derivative'] = month_df['CASH DIFF'].diff()

#     # Plot derivative on secondary y-axis
#     ax2 = ax.twinx()
#     ax2.plot(month_df['Pricing Day'], month_df['Derivative'], color='green', linestyle=':', marker='x', label='1st Derivative')
#     ax2.set_ylabel('Δ CASH DIFF', color='green')
#     ax2.tick_params(axis='y', labelcolor='green')

#     # Optional: add legend for second y-axis
#     lines, labels = ax.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper right')
#     ax.plot(month_df['Pricing Day'], month_df['CASH DIFF'], marker='o', color='blue', label='Cash Diff')
#     monthly_avg = month_df['CASH DIFF'].mean()
#     ax.axhline(monthly_avg, color='red', linestyle='--', linewidth=1, label='Monthly Avg')

#     ax.set_title(month.strftime('%b-%y'))
#     ax.set_xlabel('Pricing Day')
#     ax.set_ylabel('USD/MT')
#     ax.grid(True)
#     ax.legend(loc='upper left')

#     # Set x-axis ticks to match pricing days
#     max_day = month_df['Pricing Day'].max()
#     ax.set_xticks(range(1, max_day + 1, 2))
#     ax.set_xlim(1, max_day)
#     ax.yaxis.set_major_locator(MultipleLocator(5))      # For CASH DIFF
#     ax2.yaxis.set_major_locator(MultipleLocator(5))
# # Hide unused subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# fig.suptitle('Daily CASH DIFF Decay by Pricing Day (per Month)', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.show()
