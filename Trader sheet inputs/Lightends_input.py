import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import shutil
import os

######## INPUT ##############
ice_report = 'ICE_250729'

primary_path = f'M:/16.Gasoline/ICE_Reports/{ice_report}.xlsx'
fallback_path = f'M:/24.Naphtha/ICE_Reports/{ice_report}.xlsx'

# Check if the file exists at the primary path
if os.path.exists(primary_path):
    file_path = primary_path
elif os.path.exists(fallback_path):
    file_path = fallback_path
else:
    raise FileNotFoundError(f"{ice_report}.xlsx not found in either location.")

df_ice = pd.read_excel(file_path, header=None)

######### PREPROCESSING #######
cleared_deals_index = df_ice[df_ice.apply(lambda row: row.astype(str).str.contains('Cleared Deals').any(), axis=1)].index[0]
# Remove all rows before the "Cleared Deals" section
df_cleared = df_ice.iloc[cleared_deals_index+2:].reset_index(drop=True)
# Find the row index where the headers start (where "Trade Date" appears)
header_row_index = df_cleared[df_cleared.apply(lambda row: row.astype(str).str.contains('Trade Date').any(), axis=1)].index[0]
# Set the correct header
df_cleared.columns = df_cleared.iloc[header_row_index]
# Remove the header row from the data
df_cleared = df_cleared.iloc[header_row_index + 1:].reset_index(drop=True)
df_gas = df_cleared[df_cleared['Clearing Acct'] == '7B812']
df_nap = df_cleared[df_cleared['Clearing Acct'] == '7B814']


########### NAPHTHA ###############
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
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Naphtha Futures'),'NAP FP',expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'].str.contains('Gasoline Futures'),'Gasoline',expo_nap['Reference'])
expo_nap['Reference'] = np.where(((df_nap['Product'] == 'Naphtha Futures') & df_nap['Hub'].str.contains('Naphtha C&F')), 'MOPJ', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Product'] == 'Naphtha Futures TAPS', 'MOPJ', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC2'), 'TC2', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC5'), 'TC5', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC6'), 'TC6', expo_nap['Reference'])
expo_nap['Reference'] = np.where(df_nap['Hub'].str.contains('TC14'), 'TC14', expo_nap['Reference'])

## if crack divide by conversion factor
expo_nap['SIZE'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Nap Crk'), expo_nap['SIZE'] / 8.9, expo_nap['SIZE'])
expo_nap['SIZE'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Gas Crk'), expo_nap['SIZE'] / 8.33, expo_nap['SIZE'])
expo_nap['UNIT'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Nap Crk'), 'MT', expo_nap['UNIT'])
expo_nap['UNIT'] = np.where((expo_nap['UNIT'] == 'BBL') & (expo_nap['Reference'] == 'Gas Crk'), 'MT', expo_nap['UNIT'])

expo_nap['TRADE CLASS'] = np.where(expo_nap['Reference'] == 'EW', 'EW', expo_nap['TRADE CLASS'])

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


##################### GASOLINE ##################################
gas_cols = ['Pricing Completion 1', 'Pricing Completion 2', 'Reference', 'Hub', 'Notes', 'Trade Date', 'SAPTrade#', 'Trade Book', 'TRADE CLASS', 'PRODUCT TYPE', 'TRADE TYPE', 'PRODUCT', 'TENOR','BUY/SELL', 'SIZE', 'UNIT', 'BBL/MT', 'PRICE']
expo_gas = pd.DataFrame(columns=gas_cols).fillna('')

expo_gas['Trade Date'] = pd.to_datetime(df_gas['Trade Date'])
expo_gas['Trade Book'] = 'GASOLINE'
expo_gas['TRADE TYPE'] = 'FLATPRICE'
expo_gas['TENOR'] = pd.to_datetime(df_gas['Begin Date'])
expo_gas['BUY/SELL'] = np.where(df_gas['B/S'] == 'Bought', 'BUY', 'SELL')
expo_gas['SIZE'] = df_gas['Total Quantity']
expo_gas['UNIT'] = np.where(df_gas['Qty Units']== 'bbl', 'BBL', 'MT')
expo_gas['PRICE'] = df_gas['Price']
expo_gas['TRADE CLASS'] = np.where(df_gas['Product'].str.contains('Freight'), 'FFA', '')

expo_gas['PRODUCT TYPE'] = np.where(df_gas['Product'].str.contains('Gasoline'), 'GASOLINE', '')
expo_gas['PRODUCT TYPE'] = np.where(df_gas['Product'].str.contains('Freight'), 'GASOLINE', expo_gas['PRODUCT TYPE'])
expo_gas['PRODUCT TYPE'] = np.where(df_gas['Product'].str.contains('Naphtha'), 'NAPHTHA', expo_gas['PRODUCT TYPE'])
expo_gas['PRODUCT TYPE'] = np.where(df_gas['Product'].str.contains('Crude'), 'CRUDE', expo_gas['PRODUCT TYPE'])

expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('Argus Eurobob'), 'Eurobob', '')
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('Sing Mogas Unl'), 'MOPS_95', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('Sing Mogas 92'), 'MOPS_92', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('Premium'), 'Med_Gasoline', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Product'].str.contains('Naphtha'), 'NWE_Naphtha', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Product'].str.contains('Crude'), 'BRENT_SWAP', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('TC2'), 'TC2', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('TC5'), 'TC5', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('TC6'), 'TC6', expo_gas['PRODUCT'])
expo_gas['PRODUCT'] = np.where(df_gas['Hub'].str.contains('TC14'), 'TC14', expo_gas['PRODUCT'])

expo_gas['BBL/MT'] = np.where(expo_gas['PRODUCT TYPE'] == 'NAPHTHA', 8.9, 0)
expo_gas['BBL/MT'] = np.where(expo_gas['PRODUCT TYPE'] == 'GASOLINE', 8.33, expo_gas['BBL/MT'])
expo_gas['BBL/MT'] = np.where(expo_gas['PRODUCT TYPE'] == 'CRUDE', 7.33, expo_gas['BBL/MT'])

expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Gasoline Futures') & df_gas['Hub'].str.contains('Argus Eurobob Oxy FOB Rdam Bg'), 'GAS FP', '')
expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Naphtha Crack'), 'Nap Crk', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Gasoline Crack'), 'Gas Crk', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('Rdam Bg/Naphtha'), 'GASNAP', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Gasoline Futures') & df_gas['Hub'].str.contains('Sing Mogas 92 Unl'), '92 Ron', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'] == 'Sing Mogas 92 Unl (Platts)/Argus Eurobob Oxy FOB Rdam Bg', 'EW', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'] =='Sing Mogas 92 Unl (Platts)/Brent 1st Line', '92 Crk', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('Sing Mogas Unl 95/92'), '95/92', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('Premium Unl 10ppm FOB Med Cg'), 'Med Gasoline', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'] == 'Premium Unl 10ppm FOB Med Cg (Platts)/Argus Eurobob Oxy FOB Rdam Bg', 'MEDNTH', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'] == 'Premium Unl 10ppm FOB Med Cg (Platts)/Argus Eurobob Oxy FOB Rdam Bg Mini', 'MEDNTH', expo_gas['Reference'])

expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Crude Futures'), 'Brent', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Product'].str.contains('Naphtha Futures'), 'NAP FP', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('TC2'), 'TC2', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('TC5'), 'TC5', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('TC6'), 'TC6', expo_gas['Reference'])
expo_gas['Reference'] = np.where(df_gas['Hub'].str.contains('TC14'), 'TC14', expo_gas['Reference'])

expo_gas['SIZE'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == 'Nap Crk'), expo_gas['SIZE'] / 8.9, expo_gas['SIZE'])
expo_gas['SIZE'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == 'Gas Crk'), expo_gas['SIZE'] / 8.33, expo_gas['SIZE'])
expo_gas['SIZE'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == '92 Crk'), expo_gas['SIZE'] / 8.33, expo_gas['SIZE'])
expo_gas['UNIT'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == 'Nap Crk'), 'MT', expo_gas['UNIT'])
expo_gas['UNIT'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == 'Gas Crk'), 'MT', expo_gas['UNIT'])
expo_gas['UNIT'] = np.where((expo_gas['UNIT'] == 'BBL') & (expo_gas['Reference'] == '92 Crk'), 'MT', expo_gas['UNIT'])

expo_gas['TRADE CLASS'] = np.where(expo_gas['Reference'] == 'EW', 'EW', expo_gas['TRADE CLASS'])

conditions_to_add = ['Nap Crk', 'Gas Crk', '92 Crk', 'MEDNTH', 'EW', 'GASNAP']

gas_final = expo_gas.copy()
new_rows = []
for i in range(len(gas_final)):
    if gas_final.iloc[i]['Reference'] in conditions_to_add:
        # Create a new row to insert
        new_row = gas_final.iloc[i].copy()
        new_row['Reference'] = '' 
        new_row['PRICE'] = '' 
        new_row['PRODUCT TYPE'] = ''
        new_row['PRODUCT'] = ''

        if gas_final.iloc[i]['Reference'] == 'Nap Crk':
            new_row['PRODUCT TYPE'] = 'CRUDE'
            new_row['PRODUCT'] = 'BRENT_SWAP'
            new_row['BBL/MT'] = 7.33
        elif gas_final.iloc[i]['Reference'] == 'Gas Crk':
            new_row['PRODUCT TYPE'] = 'CRUDE'
            new_row['PRODUCT'] = 'BRENT_SWAP'
            new_row['BBL/MT'] = 7.33
        elif gas_final.iloc[i]['Reference'] == '92 Crk':
            new_row['PRODUCT TYPE'] = 'CRUDE'
            new_row['PRODUCT'] = 'BRENT_SWAP'
            new_row['BBL/MT'] = 7.33
        elif gas_final.iloc[i]['Reference'] == 'MEDNTH':
            new_row['PRODUCT TYPE'] = 'GASOLINE'
            new_row['PRODUCT'] = 'EuroBOB'
            new_row['BBL/MT'] = 8.33
        elif gas_final.iloc[i]['Reference'] == 'EW':
            new_row['PRODUCT TYPE'] = 'GASOLINE'
            new_row['PRODUCT'] = 'EuroBOB'
            new_row['BBL/MT'] = 8.33
        elif gas_final.iloc[i]['Reference'] == 'GASNAP':
            new_row['PRODUCT TYPE'] = 'NAPHTHA'
            new_row['PRODUCT'] = 'NWE_Naphtha'
            new_row['BBL/MT'] = 8.9

        if gas_final.iloc[i]['BUY/SELL'] == 'BUY':
            new_row['BUY/SELL'] = 'SELL'
        elif gas_final.iloc[i]['BUY/SELL'] == 'SELL':
            new_row['BUY/SELL'] = 'BUY'
        new_rows.append((i, new_row))
# Insert new rows into the DataFrame
for pos, row in reversed(new_rows):
    gas_final = pd.concat([gas_final.iloc[:pos+1], pd.DataFrame([row], columns=gas_final.columns), gas_final.iloc[pos+1:]]).reset_index(drop=True)


############ OUTPUT ##########################

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Add the nap_final and gas_final DataFrames as new sheets
    nap_final.to_excel(writer, sheet_name='NAP', index=False)
    gas_final.to_excel(writer, sheet_name='GAS', index=False)


# Determine where the file exists
if os.path.exists(primary_path):
    file_path = primary_path
    destination_folder = 'M:/24.Naphtha/ICE_Reports/'  # Copy to Naphtha
elif os.path.exists(fallback_path):
    file_path = fallback_path
    destination_folder = 'M:/16.Gasoline/ICE_Reports/'  # Copy to Gasoline
else:
    raise FileNotFoundError(f"{ice_report}.xlsx not found in either location.")

# Construct new file path
new_file_path = os.path.join(destination_folder, f'{ice_report}.xlsx')

# Copy the file
shutil.copy(file_path, new_file_path)

print(f"File copied from {file_path} to {new_file_path}")