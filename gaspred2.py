import pandas as pd
import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set up Eikon API
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# Define date range
end = datetime.today()
start = end - timedelta(days=365)

# If you need strings for API:
start_date = start.strftime('%Y-%m-%d')
end_date = end.strftime('%Y-%m-%d')

# Input
spread1 = ek.get_timeseries('EBOBNWEMc1', start_date=start, end_date=end)
spread2 = ek.get_timeseries('EBOBNWEMc2', start_date=start, end_date=end)
print(spread1)
spread1.dropna(inplace=True)
spread2.dropna(inplace=True)
spread = spread1- spread2
spread.dropna(axis=1, inplace=True)  # Remove NA columns
spread['50'] = spread['CLOSE'].rolling(window=50).mean()
spread['200'] = spread['CLOSE'].rolling(window=200).mean()
spread.dropna(inplace=True)

# Calculate Rolling Standard Deviation as a measure of volatility
spread['std'] = spread['CLOSE'].rolling(window=14).std()

# Calculate RSI (14-day by default)
delta = spread['CLOSE'].diff()  # Difference between consecutive prices
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
rs = gain / loss  # Relative strength
spread['RSI'] = 100 - (100 / (1 + rs))  # RSI Calculation

# Drop rows with NaN values caused by the rolling calculation
spread.dropna(inplace=True)

# Initialize variables for backtesting
spread['Position'] = 0  # 5kt for Buy, -5kt for Sell, 0 for No Position
expo = 0  # kt
expo_upper_limit = 500 # kt
expo_lower_limit = -500
allocated_kt_per_trade = 5  # kt
active_trades = []  # List to store active trades
trade_log = []  # List to store trade logs

# Initialize trade counter
trade_counter = 1

# Iterate through the spread data to simulate trades
for i in range(len(spread)):
    # Check for a buy signal (Golden Cross) or another buy condition
    if (spread['50'].iloc[i-1] < spread['200'].iloc[i-1] and spread['50'].iloc[i] > spread['200'].iloc[i] or
        spread['CLOSE'].iloc[i] < spread['CLOSE'].iloc[i-5]*0.95 and spread['RSI'].iloc[i] < 40) and expo_lower_limit <= expo <= expo_upper_limit:
        
        # Calculate buy price, stop loss, and profit target for new trade
        buy_price = spread['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
        stop_loss = buy_price - (0.8 * spread['std'].iloc[i])  # Stop loss based on volatility
        profit_target = buy_price + (1.5 * spread['std'].iloc[i])  # Profit target
        
        # Update exposure and add the new trade with trade_id
        expo += allocated_kt_per_trade  # buy so +ve
        active_trades.append({
            'trade_id': trade_counter,  # Assign a unique ID to each trade
            'buy_price': round(buy_price, 3),
            'stop_loss': round(stop_loss, 3),
            'profit_target': round(profit_target, 3),
            'buy_index': i,
            'action': 'Buy',
            'exposure': expo
        })
        trade_counter += 1  # Increment the trade_counter for the next buy
        
        spread.at[spread.index[i], 'Position'] = 5  # Mark position as active
        trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Buy', 'Price': round(float(buy_price), 2), 'Exposure': expo, 'Trade ID': trade_counter - 1})
    
    # Check for a sell signal (Golden Cross) or another sell condition
    elif (spread['50'].iloc[i-1] > spread['200'].iloc[i-1] and spread['50'].iloc[i] < spread['200'].iloc[i] or
        spread['CLOSE'].iloc[i] > spread['CLOSE'].iloc[i-5]*1.05 and spread['RSI'].iloc[i] > 70) and expo_lower_limit <= expo <= expo_upper_limit:
        
        # Calculate sell price, stop loss, and profit target for new trade
        sell_price = spread['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
        stop_loss = sell_price + (0.8 * spread['std'].iloc[i])  # Stop loss based on volatility
        profit_target = sell_price - (1.5 * spread['std'].iloc[i])  # Profit target
        
        # Update exposure and add the new trade with trade_id
        expo -= allocated_kt_per_trade  # sell so -ve
        active_trades.append({
            'trade_id': trade_counter,  # Assign a unique ID to each trade
            'sell_price': round(sell_price, 3),
            'stop_loss': round(stop_loss, 3),
            'profit_target': round(profit_target, 3),
            'sell_index': i,
            'action': 'Sell',
            'exposure': expo
        })
        trade_counter += 1  # Increment the trade_counter for the next sell
        
        spread.at[spread.index[i], 'Position'] = -5  # Mark position as active
        trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Sell', 'Price': round(float(sell_price), 2), 'Exposure': expo, 'Trade ID': trade_counter - 1})
    
    # Iterate over all active trades to check for profit-taking or stop-loss conditions
    for trade in active_trades[:]:
        # For Buy trades (Long positions)
        if trade['action'] == 'Buy':
            # Check if 2 weeks have passed since the buy (14 days)
            if i - trade['buy_index'] >= 14:
                sell_price = spread['CLOSE'].iloc[i] * 0.995  # Account for 0.5% fee
                profit = (sell_price - trade['buy_price']) * allocated_kt_per_trade # Correct profit/loss calculation
                expo -= allocated_kt_per_trade  # Update exposure with loss
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Forced sell out (15 pricing days)', 'Price': round(float(sell_price), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            # Check for stop loss or profit target
            elif spread['CLOSE'].iloc[i] <= trade['stop_loss']:  # If stop loss is hit
                profit = (spread['CLOSE'].iloc[i] - trade['buy_price']) * allocated_kt_per_trade  # Profit calculation for Buy
                expo -= allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss sell out', 'Price': round(float(spread['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            elif spread['CLOSE'].iloc[i] >= trade['profit_target']:  # If profit target is hit
                profit = (spread['CLOSE'].iloc[i] - trade['buy_price']) * allocated_kt_per_trade  # Profit calculation for Buy
                expo -= allocated_kt_per_trade  # 
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = -5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target sell out', 'Price': round(float(spread['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})

        # For Sell trades (Short positions)
        elif trade['action'] == 'Sell':
            # Check if 2 weeks have passed since the sell (14 days)
            if i - trade['sell_index'] >= 14:
                buy_price = spread['CLOSE'].iloc[i] * 1.005  # Account for 0.5% fee
                profit = (trade['sell_price'] - buy_price) * allocated_kt_per_trade # Correct profit/loss calculation
                expo += allocated_kt_per_trade  # Update exposure with loss
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Forced buy back (15 pricing days)', 'Price': round(float(buy_price), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            # Check for stop loss or profit target
            elif spread['CLOSE'].iloc[i] >= trade['stop_loss']:  # If stop loss is hit
                profit = (trade['sell_price'] - spread['CLOSE'].iloc[i]) * allocated_kt_per_trade  # Profit calculation for Sell
                expo += allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Stop Loss buy back', 'Price': round(float(spread['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})
            
            elif spread['CLOSE'].iloc[i] <= trade['profit_target']:  # If profit target is hit
                profit = (trade['sell_price'] - spread['CLOSE'].iloc[i]) * allocated_kt_per_trade  # Profit calculation for Sell
                expo += allocated_kt_per_trade  # Update exposure with profit
                active_trades.remove(trade)  # Remove trade from active trades
                spread.at[spread.index[i], 'Position'] = +5  # Mark position as closed
                trade_log.append({'Date': spread.index[i].strftime('%d-%m-%y'), 'Action': 'Profit Target buy back', 'Price': round(float(spread['CLOSE'].iloc[i]), 2), 'Profit': round(float(profit), 2), 'Exposure': expo, 'Trade ID': trade['trade_id']})

# Summarize results
total_profit = sum(trade['Profit'] for trade in trade_log if 'Profit' in trade)  # Sum of all profits
num_trades = len([trade for trade in trade_log if trade['Action'] in ['Profit Target buy back', 'Stop Loss buy back', 'Profit Target sell out', 'Stop Loss sell out', 'Forced sell out (15 pricing days)', 'Forced buy back (15 pricing days)']])

print(spread)
print(f"Profit: ${total_profit:.2f}")
print(f"Number of Trades: {num_trades}")

# Plot Buy and Sell Points
buy_signals = spread[spread['Position'] == 5]
sell_signals = spread[spread['Position'] == -5]

print("Buy Signals:\n", buy_signals[['CLOSE', 'Position']])
print("Sell Signals:\n", sell_signals[['CLOSE', 'Position']])

# Plot the graph
plt.figure(figsize=[14, 7])
plt.plot(spread['CLOSE'].index, spread['CLOSE'], label='Spread', color='blue')
plt.plot(spread['50'].index, spread['50'], label='50-Day Moving Average', color='orange')
plt.plot(spread['200'].index, spread['200'], label='200-Day Moving Average', color='green')

# Scatter plot for buy and sell signals
plt.scatter(buy_signals.index, buy_signals['CLOSE'], label='Buy Signal', marker='^', color='green', s=100)
plt.scatter(sell_signals.index, sell_signals['CLOSE'], label='Sell/Stop Loss Signal', marker='v', color='red', s=100)


# Annotate each trade on the chart
print("Trade Log:")
trade_details = []
latest_price = round(spread['CLOSE'][-1], 2)
latest_date = pd.to_datetime(spread.index[-1]).strftime('%d-%m-%y')

for trade in trade_log:
    print(trade)
    trade_date = datetime.strptime(trade['Date'], '%d-%m-%y')
    trade_price = trade['Price']
    trade_id = trade['Trade ID']

    if trade['Action'] == 'Buy':
        buy_trade = trade
        # Look for the corresponding Sell trade for this Buy trade
        sell_trade = next((t for t in trade_log if t['Trade ID'] == trade_id and ('Sell' in t['Action'] or 'Stop Loss sell out' in t['Action'] or 'Profit Target sell out' in t['Action'] or 'Forced sell out (15 pricing days)'in t['Action'])), None)

        if sell_trade:
            # If a corresponding sell trade exists, plot both buy and sell
            sell_date = datetime.strptime(sell_trade['Date'], '%d-%m-%y')
            sell_price = sell_trade['Price']

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Plot Sell trade (red)
            plt.text(sell_date, sell_price, f"{trade_id}", color='red', fontsize=9, ha='left', va='top')

            # Calculate profit for Buy trade
            profit = round((sell_price - trade_price) * allocated_kt_per_trade, 2)

            # Append trade details
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                sell_trade['Date'],
                sell_trade['Price'],
                profit,
                sell_trade['Action']
            ])
        else:
            # If no corresponding Sell trade found (in case it's an open position)
            # Calculate Mark-to-Market (Mtm) profit (latest price - buy price)
            mtm_profit = round((latest_price - trade_price) * allocated_kt_per_trade, 2)

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id} (Buy)", color='green', fontsize=9, ha='left', va='bottom')

            # Plot "Open Position" text and Mtm profit
            plt.text(trade_date, trade_price, f"Open Position", color='orange', fontsize=9, ha='left', va='top')

            # Append trade details for open position
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                latest_date,  # Use the latest date from the spread for open positions
                latest_price,  # Use latest price from the spread as the sell price for open positions
                mtm_profit,
                'Open Position'
            ])

    elif trade['Action'] == 'Sell':
        sell_trade = trade
        # Look for the corresponding Buy trade for this Sell trade
        buy_trade = next((t for t in trade_log if t['Trade ID'] == trade_id and ('Buy' in t['Action'] or 'Stop Loss buy back' in t['Action'] or 'Profit Target buy back' in t['Action'] or 'Forced buy back (15 pricing days)' in t['Action'])), None)

        if buy_trade:
            # If a corresponding buy trade exists, plot both sell and buy
            buy_date = datetime.strptime(buy_trade['Date'], '%d-%m-%y')
            buy_price = buy_trade['Price']

            # Plot Sell trade (red)
            plt.text(trade_date, trade_price, f"{trade_id}", color='red', fontsize=9, ha='left', va='top')

            # Plot Buy trade (green)
            plt.text(buy_date, buy_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Calculate profit for Sell trade
            profit = round((trade_price - buy_price) * allocated_kt_per_trade, 2)

            # Append trade details
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                buy_trade['Date'],
                buy_trade['Price'],
                sell_trade['Date'],
                sell_trade['Price'],
                profit,
                buy_trade['Action']
            ])
        else:            
            mtm_profit = round((trade_price - latest_price) * allocated_kt_per_trade, 2)

            # Plot Buy trade (green)
            plt.text(trade_date, trade_price, f"{trade_id}", color='green', fontsize=9, ha='left', va='bottom')

            # Plot "Open Position" text and Mtm profit
            plt.text(trade_date, trade_price, f"Open Position", color='orange', fontsize=9, ha='left', va='top')

            # Append trade details for open position
            trade_details.append([
                trade_id,
                allocated_kt_per_trade,
                latest_date,
                latest_price,
                sell_trade['Date'],
                sell_trade['Price'],  
                mtm_profit,
                'Open Position'
            ])

# Add legend and title
plt.legend(loc='upper left', fontsize='small')
plt.title('Gas Spread Backtest')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.grid(True)
# Add text outside the plot area (bottom left of the figure)
plt.figtext(0.95, 0.95, f"Latest Date: {latest_date}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.93, f"Latest Price: ${latest_price:.2f}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.91, f"Total Trades: {num_trades}", color='black', fontsize=8, ha='right', va='bottom')
plt.figtext(0.95, 0.89, f"Total Profit: ${total_profit*1000:.2f}", color='black', fontsize=8, ha='right', va='bottom')


# Prepare data for the table, including the reason for profit-taking
column_labels = ['Trade ID', 'Position (kt)', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Profit ($000s)', 'Action']

# Only plot the table if there are trade details
if trade_details:
    # Create the table as a separate figure
    fig, ax = plt.subplots(figsize=(8, 12))
    fig.subplots_adjust(top=0.9)
    fig.text(0.5, 0.97, 'Trade Details', ha='center', va='center', fontsize=12, fontweight='bold')
    summary_text = f"Total Profit: ${total_profit*1000:.2f}\nNumber of Trades: {num_trades}"
    fig.text(0.5, 0.95, summary_text, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.axis('off')  # Turn off the axis
    table = plt.table(cellText=trade_details, colLabels=column_labels, loc='center', cellLoc='center', colColours=['#f2f2f2'] * len(column_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(column_labels))))

    # Show the plot
    plt.show()
else:
    print("No trade details available for table display.")

