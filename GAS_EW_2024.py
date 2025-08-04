import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import eikon as ek

# File path
file_path = r'C:\Users\wangwz\OneDrive - Aramco Overseas Company\Trading\gasoline backtests\EW_PNL.xlsx'

# Load the Excel file
df_sheet = pd.read_excel(file_path, header=None)
df_sheet = df_sheet.reset_index(drop=True)
df_sheet.columns = df_sheet.iloc[0]
df_sheet = df_sheet.drop(0)
df_sheet['TENOR'] = pd.to_datetime(df_sheet['TENOR'], errors='coerce')
df_sheet['Trade Date'] = pd.to_datetime(df_sheet['Trade Date'], errors='coerce')
df_sheet['Month_Number'] = df_sheet['TENOR'].dt.month
df_sheet = df_sheet.sort_values(by='Trade Date', ascending=True)

# Eikon API Key
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

#################################################
start = '2023-06-01'
end = '2024-12-31'

# Month mappings
month_mapping = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

# Define date range
dates = pd.date_range(start=start, end=end, freq='D')

# Crack data columns
col_east = [f"MOG92SGM{code}4^2" for code in month_mapping.values()]
col_west = [f"TFSMONWEFPM{code}4^2" for code in month_mapping.values()]

# Create DataFrame for East and West data
df_east = pd.DataFrame(index=dates, columns=col_east)
df_west = pd.DataFrame(index=dates, columns=col_west)

# Fetch data from Eikon for each ticker
for ticker_east, ticker_west in zip(col_east, col_west):
    try:
        df_east[ticker_east] = ek.get_timeseries(ticker_east, start_date=start, end_date=end)['CLOSE']
        df_west[ticker_west] = ek.get_timeseries(ticker_west, start_date=start, end_date=end)['CLOSE']
    except Exception as e:
        print(f"Error fetching data for {ticker_east} or {ticker_west}: {e}")

# Calculate East-West Forward Curves for each month
df_ew = pd.DataFrame(index=dates, columns=col_east)  # Creating an empty DataFrame with East columns
for i, ticker_east in enumerate(col_east):
    ticker_west = col_west[i]
    df_ew[ticker_east] = df_east[ticker_east] - df_west[ticker_west]/8.33

# Rename columns by month for clarity
month_labels = [f"2024-{str(i+1).zfill(2)}" for i in range(12)]  # Create 12 months for 2024
df_ew.columns = month_labels  # Renaming columns to reflect the months

# Save to CSV if needed
df_ew.to_csv(r'C:\Users\wangwz\OneDrive - Aramco Overseas Company\Trading\gasoline backtests\df_forwards_ew.csv', index=True)

# Show the first few rows of the result
print(df_ew.head())

# Naptha data
def get_ew_price(row):
    trade_date = pd.to_datetime(row['Trade Date'])
    tenor_date = pd.to_datetime(row['TENOR']).strftime('%Y-%m')  # Format to match column labels
    
    if trade_date in df_ew.index and tenor_date in df_ew.columns:
        return df_ew.at[trade_date, tenor_date]
    else:
        return np.nan

# Apply functions to get prices
df_sheet['EW Price'] = df_sheet.apply(get_ew_price, axis=1)
df_sheet['EW PNL'] = np.where(df_sheet['BUY/SELL'] == 'BUY', df_sheet['SIZE'] * (df_sheet['Value'] - df_sheet['EW Price']), df_sheet['SIZE'] * (df_sheet['EW Price'] - df_sheet['Value']))
pd.set_option('display.max_columns', None)
print(df_sheet.head(50))

# Calculate total PnL for each of the columns
total_pnl = df_sheet['EW PNL'].sum()

# Print the results
print(f"Total PnL: {total_pnl}")

# Convert the 'Trade Date' column to datetime if it is not already
df_sheet['Trade Date'] = pd.to_datetime(df_sheet['Trade Date'])


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Calculate cumulative PnL for each strategy
df_sheet['Cumulative PNL'] = df_sheet['PNL'].cumsum()


# Track cumulative crack position
df_sheet['Cumulative EW Position'] = 0

# Initialize the positions for the first row
df_sheet['Cumulative EW Position'] = df_sheet['SIZE'].iloc[0]

# Update the position based on BUY/SELL
for i in range(1, len(df_sheet)):
    if df_sheet.iloc[i]['BUY/SELL'] == 'BUY':
        df_sheet.loc[df_sheet.index[i], 'Cumulative EW Position'] = df_sheet.iloc[i-1]['Cumulative EW Position'] + df_sheet.iloc[i]['SIZE']
        
    elif df_sheet.iloc[i]['BUY/SELL'] == 'SELL':
        df_sheet.loc[df_sheet.index[i], 'Cumulative EW Position'] = df_sheet.iloc[i-1]['Cumulative EW Position'] - df_sheet.iloc[i]['SIZE']
    else:
        df_sheet.loc[df_sheet.index[i], 'Cumulative EW Position'] = df_sheet.iloc[i-1]['Cumulative EW Position']
        
print('2024')

print(df_sheet)
df_sheet.to_csv(r'C:\Users\wangwz\OneDrive - Aramco Overseas Company\Trading\gasoline backtests\df_ew_2024.csv', index=True)

# Create subplots for cumulative PnL, Crack Spread, Crude Price, and Crack Position
fig, axs = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Plot cumulative PnL for each strategy
axs[0].plot(df_sheet['Trade Date'], df_sheet['Cumulative PNL'], label='Cumulative PnL', color='red')
axs[0].set_title('Cumulative PnL 2024')
axs[0].set_ylabel('Cumulative PnL ($)')
axs[0].legend()
axs[0].grid(True)

# Plot Cumulative Position
axs[1].plot(df_sheet['Trade Date'], df_sheet['Cumulative EW Position'], label='Crack Position', color='red')
axs[1].set_title('Cumulative Position (BBL)')
axs[1].set_ylabel('Cumulative Position (BBL)')
axs[1].legend()
axs[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()