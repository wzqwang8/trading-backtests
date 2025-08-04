import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set up Eikon API
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Define date range
start = '2024-01-01'
end = '2025-02-04'

# Input
brent = ek.get_timeseries('AAYES00', start_date=start, end_date=end)
brent.dropna(axis=1, inplace=True)
brent['50'] = brent['CLOSE'].rolling(window=50).mean()
brent['200'] = brent['CLOSE'].rolling(window=200).mean()
brent.dropna(inplace=True)
brent['Prev_Close'] = brent['CLOSE'].shift(1)
brent.iloc[0, brent.columns.get_loc('Prev_Close')] = brent.iloc[1, brent.columns.get_loc('Prev_Close')]

# Calculate Rolling Standard Deviation as a measure of volatility
brent['std'] = brent['CLOSE'].rolling(window=14).std()

# Drop rows with NaN values caused by the rolling calculation
brent.dropna(inplace=True)
print(brent)

# Initialize variables for backtesting
brent['Position'] = 0  # 1 for Buy, -1 for Sell, 0 for No Position
cash = 1000000  # Total cash (not fully used for each trade)
allocated_cash_per_trade = 100  # Fixed amount per trade
active_trades = []  # List to store active trades
trade_log = []

# Iterate through the brent data to simulate trades
for i in range(len(brent)):
    # Check for a buy signal (Golden Cross) or another buy condition
    if (brent['50'].iloc[i-1] < brent['200'].iloc[i-1] and brent['50'].iloc[i] > brent['200'].iloc[i] or
        brent['CLOSE'].iloc[i] < brent['CLOSE'].iloc[i-5]*0.95) and cash >= allocated_cash_per_trade:
        
        # Calculate buy price, stop loss, and profit target for new trade
        buy_price = brent['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
        stop_loss = buy_price - (2 * brent['std'].iloc[i])  # Stop loss based on volatility
        profit_target = buy_price * 1.05  # Profit target
        
        # Update cash and add the new trade
        cash -= allocated_cash_per_trade  # Subtract cash for this trade
        active_trades.append({
            'buy_price': buy_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'buy_index': i,
            'action': 'Buy'
        })
        
        brent.at[brent.index[i], 'Position'] = 1  # Mark position as active
        trade_log.append({'Date': brent.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price))})
    
    # Iterate over all active trades to check for profit-taking or stop-loss conditions
    for trade in active_trades[:]:
        # Check if 2 weeks have passed since the buy (14 days)
        if i - trade['buy_index'] >= 14:
            sell_price = brent['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += loss  # Update cash with loss
            active_trades.remove(trade)  # Remove trade from active trades
            brent.at[brent.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': brent.index[i].strftime('%d-%m-%y'), 'Action': 'Forced Close (2 Weeks)', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})
        
        # Check for profit target
        elif brent['CLOSE'].iloc[i] >= trade['profit_target']:  # If profit target is hit
            sell_price = brent['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            profit = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += profit  # Update cash with profit
            active_trades.remove(trade)  # Remove trade from active trades
            brent.at[brent.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': brent.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target', 'Price': round(float(sell_price)), 'Profit': round(float(profit))})
        
        # Check for stop loss
        elif brent['CLOSE'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
            sell_price = brent['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
            loss = (sell_price - trade['buy_price']) * (allocated_cash_per_trade / trade['buy_price'])
            cash += loss  # Update cash with loss
            active_trades.remove(trade)  # Remove trade from active trades
            brent.at[brent.index[i], 'Position'] = -1  # Mark position as closed
            trade_log.append({'Date': brent.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss', 'Price': round(float(sell_price)), 'Loss': round(float(loss))})

# Summarize results
total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
total_loss = sum(trade['Loss'] for trade in trade_log if 'Loss' in trade)  # Sum of all losses
net_profit = total_profit + total_loss  # Combine profits and losses
num_trades = len([trade for trade in trade_log if trade['Action'] == 'Profit Target' or trade['Action'] == 'Stop Loss' or trade['Action'] == 'Forced Close (2 Weeks)'])

print(f"Total Profit: ${total_profit:.2f}")
print(f"Total Loss: ${total_loss:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Number of Trades: {num_trades}")
print("Trade Log:")
for trade in trade_log:
    print(trade)

# Visualize the trades on the chart
plt.figure(figsize=[14, 7])
plt.plot(brent['CLOSE'].index, brent['CLOSE'], label='Brent', color='blue')
plt.plot(brent['50'].index, brent['50'], label='50-Day Moving Average', color='orange')
plt.plot(brent['200'].index, brent['200'], label='200-Day Moving Average', color='green')

# Plot Buy and Sell Points
buy_signals = brent[brent['Position'] == 1]
sell_signals = brent[brent['Position'] == -1]

plt.scatter(buy_signals.index, buy_signals['CLOSE'], label='Buy Signal', marker='^', color='green', s=100)
plt.scatter(sell_signals.index, sell_signals['CLOSE'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)

plt.title('Brent with Backtest - 50-Day and 200-Day Moving Averages (With Stop Loss)')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.legend()
plt.grid(True)
plt.show()
