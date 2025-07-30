import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import eikon as ek
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from datetime import timedelta
from matplotlib.cm import get_cmap

# Eikon API Key
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

#################################################
start = '2022-01-01'
end = '2025-04-30'

# Month mappings
prin_close = ek.get_timeseries('PAAAL00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
moc_close = ek.get_timeseries('PAAAJ00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})

# Calculate CASH DIFF
cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH DIFF')

# Combine into one DataFrame
df = pd.concat([prin_close, moc_close, cash_diff], axis=1)

# Drop rows with all NaNs (optional)
df.dropna(how='all', inplace=True)

# Define time ranges
end_date = pd.to_datetime("today").normalize()
date_ranges = [
    (end_date - pd.DateOffset(years=1), end_date),
    (end_date - pd.DateOffset(years=2), end_date),
    (end_date - pd.DateOffset(years=5), end_date)
]

titles = ["Past year", "Past 2 years", "Past 5 years"]

# Set up the subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
bins = [0, 5, 10, 20, float('inf')]
labels = ['0–5', '5–10', '10–20', '20+']
colors = get_cmap("tab10")

show_individual = True  # Toggle to show individual months
max_months_per_bin = 10  # Limit individual lines per bin to avoid clutter

for i, ((start_range, end_range), title) in enumerate(zip(date_ranges, titles)):
    # Filter data for the period
    df_period = df[(df.index >= start_range) & (df.index <= end_range)].copy()
    df_reset = df_period.reset_index()
    df_reset['Month'] = df_reset['Date'].dt.to_period('M')
    df_valid = df_reset.dropna(subset=['CASH DIFF']).copy()
    df_valid['Pricing Day'] = df_valid.groupby('Month').cumcount() + 1

    # Calculate derivative and abs derivative
    df_valid['Derivative'] = df_valid.groupby('Month')['CASH DIFF'].diff()
    df_valid = df_valid.dropna(subset=['Derivative'])
    df_valid['Abs Derivative'] = df_valid['Derivative'].abs()

    # Bin by average CASH DIFF per month
    avg_by_month = df_valid.groupby('Month')['CASH DIFF'].mean()
    df_valid['Diff Bin'] = df_valid['Month'].map(pd.cut(avg_by_month, bins=bins, labels=labels))

    # Plot each bin
    for j, (label, group_df) in enumerate(df_valid.groupby('Diff Bin')):
        # --- Individual month lines ---
        if show_individual:
            for k, (month, month_df) in enumerate(group_df.groupby('Month')):
                if k >= max_months_per_bin:
                    break  # Limit how many individual lines are shown
                individual_cash = month_df.groupby('Pricing Day')['CASH DIFF'].mean()
                axs[i].plot(individual_cash.index, individual_cash.values,
                            color=colors(j), alpha=0.2, linewidth=0.8)

        # --- Average and bands ---
        avg_cash_diff = group_df.groupby('Pricing Day')['CASH DIFF'].mean()
        std_cash_diff = group_df.groupby('Pricing Day')['CASH DIFF'].std()
        avg_abs_derivative = group_df.groupby('Pricing Day')['Abs Derivative'].mean()

        axs[i].fill_between(avg_cash_diff.index,
                            avg_cash_diff - std_cash_diff,
                            avg_cash_diff + std_cash_diff,
                            color=colors(j), alpha=0.1)

        axs[i].plot(avg_cash_diff.index, avg_cash_diff.values,
                    linestyle='--', marker='x',
                    label=f'{label} Avg CASH DIFF', color=colors(j), linewidth=2)
        axs[i].plot(avg_abs_derivative.index, avg_abs_derivative.values,
                    linestyle='-', marker='o',
                    label=f'{label} Avg |Δ|', color=colors(j), linewidth=2, alpha=0.5)

    # Final plot setup per subplot
    axs[i].set_title(title)
    axs[i].set_xlabel('Pricing Day')
    axs[i].set_xticks(range(1, df_valid['Pricing Day'].max() + 1))
    axs[i].grid(True)
    axs[i].legend(fontsize=8)

axs[0].set_ylabel('$/mt')
plt.suptitle('Avg CASH DIFF and |Δ CASH DIFF| by Pricing Day\nGrouped by Avg CASH DIFF Bin', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



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