import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time


ice_report = 'ICE_082724' ### chose ice report

# Load the Excel file
file_path = f'C:/Users/wangwz/OneDrive - Aramco Overseas Company/Trading/ICE_Reports/{ice_report}.xlsx'
df_ice = pd.read_excel(file_path, header=None)

## Preprocessing of ice report
cleared_deals_index = df_ice[df_ice.apply(lambda row: row.astype(str).str.contains('Cleared Deals').any(), axis=1)].index[0]
# Remove all rows before the "Cleared Deals" section
df_cleared = df_ice.iloc[cleared_deals_index+2:].reset_index(drop=True)
# Find the row index where the headers start (where "Trade Date" appears)
header_row_index = df_cleared[df_cleared.apply(lambda row: row.astype(str).str.contains('Trade Date').any(), axis=1)].index[0]
# Set the correct header
df_cleared.columns = df_cleared.iloc[header_row_index]
# Remove the header row from the data
df_cleared = df_cleared.iloc[header_row_index + 1:].reset_index(drop=True)

df_nap = df_cleared[df_cleared['Clearing Acct'] == '7B814']
print(df_nap)

 ## Creating expo input
nap_cols = ['1640', 'Pricing Completion 1', 'Pricing Completion 2', 'Reference', 'Hub', 'Notes', 'Trade Date', 'SAPTrade#', 'Trade Book', 'TRADE CLASS', 'PRODUCT TYPE', 'TRADE TYPE', 'PRODUCT', 'TENOR','BUY/SELL', 'SIZE', 'UNIT', 'BBL/MT', 'PRICE']
expo_nap = pd.DataFrame(columns=nap_cols).fillna('')

expo_nap['Trade Date'] = pd.to_datetime(df_nap['Trade Date'])
expo_nap['Trade Book'] = 'NAPHTHA'
expo_nap['TRADE TYPE'] = 'FLATPRICE'
expo_nap['TENOR'] = pd.to_datetime(df_nap['Begin Date'], format='%d/%m/%Y')
expo_nap['BUY/SELL'] = np.where(df_nap['B/S'] == 'Bought', 'BUY', 'SELL')
expo_nap['SIZE'] = df_nap['Total Quantity']
expo_nap['UNIT'] = np.where(df_nap['Qty Units']== 'bbl', 'BBL', 'MT')
expo_nap['PRICE'] = df_nap['Price']
expo_nap['TRADE CLASS'] = np.where(df_nap['Product'].str.contains('Freight'), 'FFA', '')

expo_nap['PRODUCT TYPE'] = np.where(df_nap['Product'].str.contains('Naphtha'), 'NAPHTHA', '')
expo_nap['PRODUCT TYPE'] = np.where(df_nap['Product'].str.contains('Freight'), 'NAPHTHA', expo_nap['PRODUCT TYPE'])
expo_nap['PRODUCT TYPE'] = np.where(df_nap['Product'].str.contains('Gasoline'), 'GASOLINE', expo_nap['PRODUCT TYPE'])
expo_nap['PRODUCT TYPE'] = np.where(df_nap['Product'].str.contains('Crude'), 'CRUDE', expo_nap['PRODUCT TYPE'])

expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('Naphtha CIF'), 'NWE_Naphtha', '')
expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('Naphtha C&F'), 'MOPJ_Naphtha', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Product'].str.contains('Gasoline'), 'EuroBOB', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Product'].str.contains('Crude'), 'BRENT_SWAP', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('TC2'), 'TC2', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('TC5'), 'TC5', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('TC6'), 'TC6', expo_nap['PRODUCT'])
expo_nap['PRODUCT'] = np.where(df_nap['Hub'].str.contains('TC14'), 'TC14', expo_nap['PRODUCT'])

expo_nap['BBL/MT'] = np.where(expo_nap['PRODUCT TYPE'] == 'NAPHTHA', 8.9, 0)
expo_nap['BBL/MT'] = np.where(expo_nap['PRODUCT TYPE'] == 'GASOLINE', 8.33, expo_nap['BBL/MT'])
expo_nap['BBL/MT'] = np.where(expo_nap['PRODUCT TYPE'] == 'CRUDE', 7.33, expo_nap['BBL/MT'])

expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Naphtha Crack'), 'Nap Crk', '')
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Gasoline Crack'), 'Gas Crk', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Gasoline Diff Futures'), 'GASNAP', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Naphtha Diff Futures'), 'EW', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Crude Futures'), 'Brent', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Naphtha Futures'), 'NAP FP', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC2'), 'TC2', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC5'), 'TC5', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC6'), 'TC6', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC14'), 'TC14', expo_nap['Reference'])

## if crack divide by conversion factor
expo_nap['SIZE'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Nap Crk'), expo_nap['SIZE'] / 8.9, expo_nap['SIZE'])
expo_nap['SIZE'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Gas Crk'), expo_nap['SIZE'] / 8.33, expo_nap['SIZE'])
expo_nap['UNIT'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Nap Crk'), 'MT', expo_nap['UNIT'])
expo_nap['UNIT'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Gas Crk'), 'MT', expo_nap['UNIT'])

 ## Trades with two expos
conditions_to_add = ['Nap Crk', 'Gas Crk', 'EW', 'GASNAP']

nap_final = expo_nap.copy()

new_rows = []

for i in range(len(nap_final)):
    if nap_final.iloc[i]['Reference'] in conditions_to_add:
        # Create a new row to insert
        new_row = nap_final.iloc[i].copy()
        new_row['Reference'] = '' 
        new_row['PRICE'] = '' 
        new_row['PRODUCT TYPE'] = ''
        new_row['PRODUCT'] = ''

        if nap_final.iloc[i]['Reference'] == 'Nap Crk':
            new_row['PRODUCT TYPE'] = 'CRUDE'
            new_row['PRODUCT'] = 'BRENT_SWAP'
            new_row['BBL/MT'] = 7.33
        elif nap_final.iloc[i]['Reference'] == 'Gas Crk':
            new_row['PRODUCT TYPE'] = 'CRUDE'
            new_row['PRODUCT'] = 'BRENT_SWAP'
            new_row['BBL/MT'] = 7.33
        elif nap_final.iloc[i]['Reference'] == 'EW':
            new_row['PRODUCT TYPE'] = 'NAPHTHA'
            new_row['PRODUCT'] = 'NWE_Naphtha'
            new_row['BBL/MT'] = 8.9
        elif nap_final.iloc[i]['Reference'] == 'GASNAP':
            new_row['PRODUCT TYPE'] = 'NAPHTHA'
            new_row['PRODUCT'] = 'NWE_Naphtha'
            new_row['BBL/MT'] = 8.9
        
        if nap_final.iloc[i]['BUY/SELL'] == 'BUY':
            new_row['BUY/SELL'] = 'SELL'
        elif nap_final.iloc[i]['BUY/SELL'] == 'SELL':
            new_row['BUY/SELL'] = 'BUY'

        new_rows.append((i, new_row))

# Adding new rows
for pos, row in reversed(new_rows):
    nap_final = pd.concat([nap_final.iloc[:pos+1], pd.DataFrame([row], columns=nap_final.columns), nap_final.iloc[pos+1:]]).reset_index(drop=True)

print(nap_final) 

excel_file_path = f'M:/24.Naphtha/Python scripts/Expo_inputs/NAP_{ice_report}.xlsx'
nap_final.to_excel(excel_file_path, index=False)
