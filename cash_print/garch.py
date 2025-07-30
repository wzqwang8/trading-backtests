import warnings
import pandas as pd
import matplotlib.pyplot as plt
import eikon as ek
from arch import arch_model

# Configurations
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
ek.set_app_key('1e01b72982374e88971ea95fe42801910d7207ef')

# 1. Load and preprocess data
def load_and_preprocess_data(start_date, end_date):
    print("Loading price data from Eikon...")
    prin_close = ek.get_timeseries('PAAAL00', start_date=start_date, end_date=end_date)[['CLOSE']].rename(columns={'CLOSE': 'PRINT'})
    moc_close = ek.get_timeseries('PAAAJ00', start_date=start_date, end_date=end_date)[['CLOSE']].rename(columns={'CLOSE': 'MOC'})
    
    cash_diff = (prin_close['PRINT'] - moc_close['MOC']).to_frame(name='CASH_DIFF')
    return cash_diff

# Call the function with your date range
start_date = '2023-01-01'
end_date = '2025-06-01'
cash_diff = load_and_preprocess_data(start_date, end_date)

series = cash_diff['CASH_DIFF'].dropna()

# Instantiate the GARCH(1,1) model (mean='Zero' assumes zero mean process)
garch_model = arch_model(series, vol='Garch', p=1, q=1, mean='Zero', dist='normal')

# Fit the model
garch_result = garch_model.fit(update_freq=10)

print(garch_result.summary())

# Plot conditional volatility
plt.figure(figsize=(10, 6))
plt.plot(garch_result.conditional_volatility)
plt.title('Estimated Conditional Volatility from GARCH(1,1) Model')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()
