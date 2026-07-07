import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import eikon as ek
from cash_print_config import configure_eikon, data_path, today_iso

pd.set_option('display.max_columns', None)
configure_eikon(ek)

# -------------------------
# Parameters & Data Fetching
# -------------------------
start = '2021-01-01'
end = today_iso()

# Fetch time series from Eikon
prin_close = ek.get_timeseries('PAAAL00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
moc_close = ek.get_timeseries('PAAAJ00', start_date=start, end_date=end)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})

# Calculate CASH DIFF
cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH DIFF')
cash_diff.index = pd.to_datetime(cash_diff.index)

# -------------------------
# Feature Engineering
# -------------------------
cash_diff['Month'] = cash_diff.index.to_period('M')
cash_diff['Day'] = cash_diff.index.day
cash_diff['Half'] = np.where(cash_diff['Day'] <= 15, 'First', 'Second')

# Average CASH DIFF per half-month and full month average
half_avg = cash_diff.groupby(['Month', 'Half'])['CASH DIFF'].mean().unstack()
monthly_avg = cash_diff.groupby('Month')['CASH DIFF'].mean()

# Combine into summary dataframe
df_summary = half_avg.copy()
df_summary['Full Month Avg Cash Diff'] = monthly_avg
df_summary = df_summary.reset_index().rename(columns={
    'Month': 'Date',
    'First': 'First Half Avg Cash Diff',
    'Second': 'Second Half Avg Cash Diff'
})

# Calculate Raw Decay and Ratio
df_summary['Raw Decay'] = df_summary['First Half Avg Cash Diff'] - df_summary['Second Half Avg Cash Diff']
df_summary['Ratio'] = df_summary['Raw Decay'] / df_summary['Full Month Avg Cash Diff'].abs()

# Additional date columns
df_summary['Year'] = df_summary['Date'].dt.year
df_summary['Month'] = df_summary['Date'].dt.month
df_summary['MonthStr'] = df_summary['Date'].dt.strftime('%b')

# -------------------------
# Visualizations
# -------------------------
# Histogram of Raw Decay with labels
plt.figure(figsize=(12, 6))
bin_edges = np.arange(np.floor(df_summary['Raw Decay'].min()), np.ceil(df_summary['Raw Decay'].max()) + 2, 2)
sns.histplot(df_summary['Raw Decay'], bins=bin_edges, kde=True, color='lightblue', edgecolor='black')

for i, row in df_summary.iterrows():
    plt.text(row['Raw Decay'], 0.3 + (i % 10) * 0.15, row['MonthStr'] + '-' + str(row['Year']), 
             rotation=45, ha='center', va='bottom', fontsize=9)

plt.axvline(0, color='red', linestyle='--')
plt.title('Distribution of Raw Decay (First Half - Second Half Cash Diff)')
plt.xlabel('Raw Decay ($/mt)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Time series plot of Raw Decay and Full Month Avg Cash Diff
plt.figure(figsize=(12, 5))
month_labels = df_summary['Date'].dt.to_timestamp().dt.strftime('%b-%y')

plt.plot(month_labels, df_summary['Raw Decay'], marker='o', color='darkblue', label='1st - 2nd Half')
plt.plot(month_labels, df_summary['Full Month Avg Cash Diff'], marker='s', color='orange', label='Full Month Avg Cash Diff')

plt.title('(1st - 2nd Half) and Full Month Avg Cash Diff')
plt.xlabel('Month')
plt.ylabel('Value ($/mt)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Regression plot: Raw Decay vs Full Month Avg
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df_summary,
    x='Full Month Avg Cash Diff',
    y='Raw Decay',
    scatter_kws={'s': 60, 'alpha': 0.7},
    line_kws={'color': 'red', 'label': 'Line of Best Fit'}
)
plt.title('(1st vs 2nd Half) vs Full Month')
plt.xlabel('Full Month Avg Cash Diff ($/mt)')
plt.ylabel('1st - 2nd Half ($/mt)')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Seasonal plot of Raw Decay by month for each year (highlight 2025)
plt.figure(figsize=(12, 6))
years = sorted(df_summary['Year'].unique())
palette = sns.color_palette('tab10', len(years))
legend_handles = []

for i, year in enumerate(years):
    data = df_summary[df_summary['Year'] == year]
    if year == 2025:
        line, = plt.plot(data['Month'], data['Raw Decay'], label=str(year), color='black', linewidth=2.5, marker='o')
    else:
        line, = plt.plot(data['Month'], data['Raw Decay'], label=str(year), color=palette[i], linewidth=1.5, marker='o')
    legend_handles.append(line)

plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Seasonal Plot of 1st - 2nd Half Raw Decay by Month and Year')
plt.xlabel('Month')
plt.ylabel('1st - 2nd Half ($/mt)')
plt.grid(True)
plt.legend(handles=legend_handles, title='Year')
plt.tight_layout()
plt.show()

# Heatmap of Raw Decay by Year and Month
pivot_heatmap = df_summary.pivot(index='Year', columns='Month', values='Raw Decay').astype(float)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_heatmap, annot=True, cmap='RdBu_r', center=0, fmt=".1f")
plt.title('Heatmap of 1st half vs 2nd half Raw Decay by Year and Month')
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()
plt.show()

# -------------------------
# Clustering on Raw Decay
# -------------------------
features = df_summary[['Raw Decay']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df_summary['Cluster'] = kmeans.fit_predict(scaled)

# Pairplot of half cash diffs colored by cluster
sns.pairplot(df_summary, vars=['First Half Avg Cash Diff', 'Second Half Avg Cash Diff'], 
             hue='Cluster', palette='Set2')
plt.suptitle("Clustered First/Second Half Behaviors (by Raw Decay)", y=1.02)
plt.tight_layout()
plt.show()

# Cluster centers summary (excluding Raw Decay)
cluster_summary = df_summary.groupby('Cluster')[[
    'First Half Avg Cash Diff',
    'Second Half Avg Cash Diff',
    'Full Month Avg Cash Diff'
]].mean().round(2)
print("Cluster Centers (mean values):")
print(cluster_summary)

# Show months grouped by cluster
months_per_cluster = df_summary[['Date', 'MonthStr', 'Year', 'Raw Decay', 'Cluster']].sort_values(by='Cluster')
print("\nMonths by Cluster:")
print(months_per_cluster)

# Boxplot of Raw Decay by cluster
plt.figure(figsize=(10, 5))
sns.boxplot(x='Cluster', y='Raw Decay', data=df_summary, palette='Set2')
plt.title('Distribution of Raw Decay by Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Monthly distribution of clusters
plt.figure(figsize=(12, 6))
sns.countplot(x='MonthStr', hue='Cluster', data=df_summary, palette='Set2')
plt.title('Monthly Distribution of Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

# PCA visualization of clusters (1D feature but still shown)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
df_summary['PC1'] = pca_result[:, 0]
df_summary['PC2'] = pca_result[:, 1]

sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_summary, palette='Set2')
plt.title("Clusters in PCA space (based on Raw Decay)")
plt.grid(True)
plt.tight_layout()
plt.show()
