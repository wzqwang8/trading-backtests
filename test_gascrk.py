import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set up Eikon API
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Define date range
start = '2023-10-01'
end = '2025-02-04'

# Input
crack1 = ek.get_timeseries('EBOBNWECKMc1', start_date=start, end_date=end)
crack2 = ek.get_timeseries('EBOBNWECKMc2', start_date=start, end_date=end)
print(crack1)
crack1.dropna(inplace=True)
crack2.dropna(inplace=True)
crack_roll = crack1
crack_roll.dropna(axis=1, inplace=True)  # Remove NA columns
crack_roll['50'] = crack_roll['CLOSE'].rolling(window=50).mean()
crack_roll['200'] = crack_roll['CLOSE'].rolling(window=200).mean()
crack_roll.dropna(inplace=True)

# Calculate Rolling Standard Deviation as a measure of volatility
crack_roll['std'] = crack_roll['CLOSE'].rolling(window=14).std()

# Calculate RSI (14-day by default)
delta = crack_roll['CLOSE'].diff()  # Difference between consecutive prices
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
rs = gain / loss  # Relative strength
crack_roll['RSI'] = 100 - (100 / (1 + rs))  # RSI Calculation

# Drop rows with NaN values caused by the rolling calculation
crack_roll.dropna(inplace=True)

# Initialize variables for backtesting
crack_roll['Position'] = 0  # 5 for Buy, -5 for Sell, 0 for No Position
exposure = 1000  # Total exposure in KT
allocated_kt_per_trade = 5  # Quantity per trade in KT
active_trades = []  # List to store active trades
trade_log = []

# Initialize trade counter
trade_counter = 1

# Iterate through the spread data to simulate trades
for i in range(len(crack_roll)):
    # Check for a buy signal (Golden Cross) or another buy condition
    if (crack_roll['50'].iloc[i-1] < crack_roll['200'].iloc[i-1] and crack_roll['50'].iloc[i] > crack_roll['200'].iloc[i] or
        crack_roll['CLOSE'].iloc[i] < crack_roll['CLOSE'].iloc[i-5]*0.95 and crack_roll['RSI'].iloc[i] < 40) and exposure >= allocated_kt_per_trade:
        
        # Calculate buy price, stop loss, and profit target for new trade
        buy_price = crack_roll['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
        stop_loss = buy_price - (1 * crack_roll['std'].iloc[i])  # Stop loss based on volatility
        profit_target = buy_price + (1.2 * crack_roll['std'].iloc[i])   # Profit target
        
        # Update exposure and add the new trade with trade_id
        exposure -= allocated_kt_per_trade  # Subtract exposure for this trade
        active_trades.append({
            'trade_id': trade_counter,  # Assign a unique ID to each trade
            'buy_price': round(buy_price, 3),
            'stop_loss': round(stop_loss, 3),
            'profit_target': round(profit_target, 3),
            'buy_index': i,
            'action': 'Buy',
            'exposure': exposure
        })
        trade_counter += 1  # Increment the trade_counter for the next buy
        
        crack_roll.at[crack_roll.index[i], 'Position'] = 5  # Mark position as active
        trade_log.append({'Date': crack_roll.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price), 2), 'Exposure': exposure, 'Trade ID': trade_counter - 1})
    
    # Iterate over all active trades to check for profit-taking or stop-loss conditions
    for trade in active_trades[:]:
        # Check if 2 weeks have passed since the buy (14 days)
        if i - trade['buy_index'] >= 15:
            sell_price = crack_roll['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            profit = (sell_price - trade['buy_price']) * allocated_kt_per_trade # Correct profit/loss calculation
            exposure += allocated_kt_per_trade  # Update exposure with loss
            active_trades.remove(trade)  # Remove trade from active trades
            crack_roll.at[crack_roll.index[i], 'Position'] = -5  # Mark position as closed
            trade_log.append({'Date': crack_roll.index[i].strftime('%d-%m-%y'), 'Action': 'Forced Close (15 pricing days)', 'Price': round(float(sell_price), 2), 'Profit': round(float(profit), 2), 'Exposure': exposure, 'Trade ID': trade['trade_id']})
        
        # Check for profit target
        elif crack_roll['CLOSE'].iloc[i] >= trade['profit_target']:  # If profit target is hit
            sell_price = crack_roll['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            profit = (sell_price - trade['buy_price']) * allocated_kt_per_trade  # Correct profit/loss calculation
            exposure += allocated_kt_per_trade  # Update exposure with profit
            active_trades.remove(trade)  # Remove trade from active trades
            crack_roll.at[crack_roll.index[i], 'Position'] = -5  # Mark position as closed
            trade_log.append({'Date': crack_roll.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target', 'Price': round(float(sell_price), 2), 'Profit': round(float(profit), 2), 'Exposure': exposure, 'Trade ID': trade['trade_id']})
        
        # Check for stop loss
        elif crack_roll['CLOSE'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
            sell_price = crack_roll['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            profit = (sell_price - trade['buy_price']) * allocated_kt_per_trade  # Correct profit/loss calculation
            exposure += allocated_kt_per_trade  # Update exposure with loss
            active_trades.remove(trade)  # Remove trade from active trades
            crack_roll.at[crack_roll.index[i], 'Position'] = -5  # Mark position as closed
            trade_log.append({'Date': crack_roll.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss', 'Price': round(float(sell_price), 2), 'Profit': round(float(profit), 2), 'Exposure': exposure, 'Trade ID': trade['trade_id']})

# Summarize results
total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
num_trades = len([trade for trade in trade_log if trade['Action'] in ['Profit Target', 'Stop Loss', 'Forced Close (15 pricing days)']])

print(crack_roll)
print(f"Profit: ${total_profit:.2f}")
print(f"Number of Trades: {num_trades}")
print("Trade Log:")

for trade in trade_log:
    print(trade)

# Plot Buy and Sell Points
buy_signals = crack_roll[crack_roll['Position'] == 5]
sell_signals = crack_roll[crack_roll['Position'] == -5]

print("Buy Signals:\n", buy_signals[['CLOSE', 'Position']])
print("Sell Signals:\n", sell_signals[['CLOSE', 'Position']])

# Plot the graph
plt.figure(figsize=[14, 7])
plt.plot(crack_roll['CLOSE'].index, crack_roll['CLOSE'], label='Crack', color='blue')
plt.plot(crack_roll['50'].index, crack_roll['50'], label='50-Day Moving Average', color='orange')
plt.plot(crack_roll['200'].index, crack_roll['200'], label='200-Day Moving Average', color='green')

# Scatter plot for buy and sell signals
plt.scatter(buy_signals.index, buy_signals['CLOSE'], label='Buy Signal', marker='^', color='green', s=100)
plt.scatter(sell_signals.index, sell_signals['CLOSE'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)

# Annotate each trade on the chart
for trade in trade_log:
    trade_date = datetime.strptime(trade['Date'], '%d-%m-%y')
    trade_price = trade['Price']
    trade_id = trade['Trade ID']

    if trade['Action'] == 'Buy':
        plt.text(trade_date, trade_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')
    else:
        plt.text(trade_date, trade_price, f"{trade_id}", color='red', fontsize=9, ha='left', va='top')

# Add legend and titles
plt.legend(loc='upper left', fontsize='small')
plt.title('Gasoline Crack Backtest')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.grid(True)

# Prepare data for the table, including the reason for profit-taking
column_labels = ['Trade ID', 'Position (kt)', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit ($000s)', 'Action']
trade_details = []

for trade in trade_log:
    if trade['Action'] in ['Profit Target', 'Stop Loss', 'Forced Close (15 pricing days)']:
        buy_trade = next(t for t in trade_log if t['Trade ID'] == trade['Trade ID'] and t['Action'] == 'Buy')
        reason_for_profit_taking = trade['Action']  # Use the 'Action' as the reason
        trade_details.append([
            trade['Trade ID'],
            allocated_kt_per_trade,
            buy_trade['Date'],
            buy_trade['Price'],
            trade['Date'],
            trade['Price'],
            round(trade.get('Profit', 0), 2),
            reason_for_profit_taking  # Adding reason for profit-taking
        ])

# Create the table as a separate figure
fig, ax = plt.subplots(figsize=(8, 9))
fig.subplots_adjust(top=0.9)
fig.text(0.5, 0.95, 'Trade Details', ha='center', va='center', fontsize=14, fontweight='bold')
summary_text = f"Total Profit: ${total_profit*1000:.2f}\nNumber of Trades: {num_trades}"
fig.text(0.5, 0.92, summary_text, ha='center', va='center', fontsize=10, fontweight='bold')
ax.axis('off')  # Turn off the axis
table = plt.table(cellText=trade_details, colLabels=column_labels, loc='center', cellLoc='center', colColours=['#f2f2f2'] * len(column_labels))
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width(col=list(range(len(column_labels))))

# Show the plot
plt.show()

