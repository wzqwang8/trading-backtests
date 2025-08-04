import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import eikon as ek
import calendar 
pd.set_option('display.max_columns', None)
# File path
file_path = r'C:\Users\wangwz\OneDrive - Aramco Overseas Company\Trading\gasoline backtests\df_ew_2024.csv'

# Load the Excel file
df_sheet = pd.read_csv(file_path, skipinitialspace=True, header=0)
df_sheet['Trade Date'] = pd.to_datetime(df_sheet['Trade Date'], errors='coerce', dayfirst=True)
df_sheet['TENOR'] = pd.to_datetime(df_sheet['TENOR'], errors='coerce', dayfirst=True)
df_sheet['Month_Number'] = df_sheet['TENOR'].dt.month
df_sheet = df_sheet.sort_values(by='Trade Date', ascending=True)
df_sheet.set_index('Trade Date', inplace=True)

print(df_sheet.head(5))
# Eikon API Key
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

#################################################
complete = []
for mon in range(2, 13):
    df_sheet1 = df_sheet[df_sheet['TENOR'].dt.month == mon]
    
    # Check if df_sheet1 is empty
    if df_sheet1.empty:
        print(f"No data for month {mon}")
        continue  # Skip this iteration if no data
    
    #################################################
    start = '2023-12-01'
    end = '2024-12-31'

    # Month mappings
    month_mapping = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

    # Dynamically get the contract code based on the selected month
    contract_letter = month_mapping[mon]
    contract_year = '4'  # Assuming 2024 contract year
    contract_suffix = '^2'

    # Construct contract symbols
    west_symbol = f'TFSMONWEFPM{contract_letter}{contract_year}{contract_suffix}'
    east_symbol = f'MOG92SGM{contract_letter}{contract_year}{contract_suffix}'

    # Fetch data dynamically
    month_west = ek.get_timeseries(west_symbol, fields='CLOSE', start_date=start, end_date=end)  
    month_east = ek.get_timeseries(east_symbol, fields='CLOSE', start_date=start, end_date=end)  
    if month_west.empty or month_east.empty:
        print(f"Skipping month {mon} due to missing data")
        print(month_east.index)
        continue  # Skip this iteration if data is missing
    month_ew = month_east - abs(month_west)/8.33

    # Merge into a combined DataFrame
    df_combined = pd.DataFrame({
        'West': month_west['CLOSE'],
        'East': month_east['CLOSE'],
        'EW': month_ew['CLOSE']
    }).dropna()
    df_combined_october = df_combined[df_combined.index.month == 10]

    # Print the data for October
    print("October Data in df_combined:")
    print(df_combined_october)


    print('df_combined')
    print(df_combined)

    # Sort values and proceed with your calculations
    df_sheet1 = df_sheet1.sort_values(by='Trade Date', ascending=True)
    print(df_sheet1.head())

    # Calculate total PnL for each of the columns
    total_pnl = df_sheet1['PNL'].sum()
    print(f"Total PnL for month {mon}: {total_pnl}")

    # Calculate cumulative PnL for each strategy
    df_sheet1['Cumulative PNL'] = df_sheet1['PNL'].cumsum()

    # Initialize the positions for the first row
    df_sheet1['Cumulative Position'] = df_sheet1['SIZE'].iloc[0]

    # Update the position based on BUY/SELL
    for i in range(1, len(df_sheet1)):
        if df_sheet1.iloc[i]['BUY/SELL'] == 'BUY':
            df_sheet1.loc[df_sheet1.index[i], 'Cumulative Position'] = df_sheet1.iloc[i-1]['Cumulative Position'] + df_sheet1.iloc[i]['SIZE']
        elif df_sheet1.iloc[i]['BUY/SELL'] == 'SELL':
            df_sheet1.loc[df_sheet1.index[i], 'Cumulative Position'] = df_sheet1.iloc[i-1]['Cumulative Position'] - df_sheet1.iloc[i]['SIZE']
        else:
            df_sheet1.loc[df_sheet1.index[i], 'Cumulative Position'] = df_sheet1.iloc[i-1]['Cumulative Position']


    fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # Get month name
    month_name = calendar.month_name[mon]

    # Plot cumulative PnL for each strategy
    axs[0].plot(df_sheet1.index, df_sheet1['Cumulative PNL'], color='black')
    axs[0].set_title('Cumulative PnL 2024')
    axs[0].set_ylabel('Cumulative PnL ($)')
    axs[0].grid(True)

    # Plot Cumulative Position
    axs[1].plot(df_sheet1.index, df_sheet1['Cumulative Position'], label='Position (BBL)', color='green')
    axs[1].set_title('Cumulative Position Over Time (2024)')
    axs[1].set_ylabel('Cumulative Position (BBL)')
    axs[1].grid(True)

    first_trade_date = df_sheet1.index.min()
    df_sheet1 = df_sheet1.loc[df_sheet1.index >= first_trade_date]
    df_combined = df_combined[df_combined.index >= first_trade_date]
    # Plot EW Spread prices for 2024
    axs[2].plot(df_sheet1.index, df_sheet1['PRICE'], label=f'Traded {month_name} EW over time', color='orange')
    axs[2].plot(df_combined.index, df_combined['EW'], label=f'Reuters {month_name} EW over time', color='blue')
    axs[2].set_title(f'{month_name} EW over time')
    axs[2].set_ylabel(f'{month_name} EW Price')
    axs[2].axhline(y=df_sheet1['Value'].mean(), color='red', linestyle='--', label=f'Final {month_name} EW')
    # Ensure POS correctly represents buy/sell size
    df_sheet1['POS'] = df_sheet1.apply(lambda row: -row['SIZE'] if row['BUY/SELL'] == 'SELL' else row['SIZE'], axis=1)

    # Group by Trade Date to get net position
    net_signals = df_sheet1.groupby('Trade Date')['POS'].sum().reset_index()

    # Separate net buys and sells
    buy_signals = net_signals[net_signals['POS'] > 0]
    sell_signals = net_signals[net_signals['POS'] < 0]

    # Get a single EW Price per trade date (e.g., using the first available price)
    EW_prices = df_sheet1.groupby('Trade Date')['PRICE'].first().reset_index()

    # Merge to ensure x and y sizes match
    buy_signals = buy_signals.merge(EW_prices, on='Trade Date', how='left')
    sell_signals = sell_signals.merge(EW_prices, on='Trade Date', how='left')

    # Plot buy and sell signals on EW Price chart
    axs[2].scatter(buy_signals['Trade Date'], buy_signals['PRICE'], marker='^', color='green', label='Net Buy Signal', zorder=5)
    axs[2].scatter(sell_signals['Trade Date'], sell_signals['PRICE'], marker='v', color='red', label='Net Sell Signal', zorder=5)
    axs[2].legend()
    axs[2].grid(True)

    plt.savefig(rf'C:\Users\wangwz\OneDrive - Aramco Overseas Company\Trading\gasoline backtests\{mon}_ew.png', bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()

months = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(months, complete, marker='o', color='b')


# Add titles and labels
plt.title('2024 PnL')
plt.xlabel('Month')
plt.ylabel('PnL $')
plt.grid(True)
# Show the plot
plt.show()