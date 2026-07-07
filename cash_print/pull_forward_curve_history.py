import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import eikon as ek
from cash_print_config import configure_eikon, data_path, today_iso
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
pd.set_option('display.max_columns', None)

# Eikon API Key
configure_eikon(ek)


dates = pd.date_range("2024-01-01", "2024-03-01", freq="B")  # Business days

all_curves = []

for d in dates:
    date_str = d.strftime('%Y-%m-%d')
    try:
        print(f"Fetching {date_str}")
        fwd_data, err = ek.get_data(
            instruments="0#NAF-NWE:",
            fields=["TRDPRC_1", "CF_DATE"],
            parameters={
                "SDate": date_str,
                "EDate": date_str
            }
        )

        if not fwd_data.empty:
            fwd_data["RunDate"] = date_str
            all_curves.append(fwd_data)

    except Exception as e:
        print(f"Error on {date_str}: {e}")

df_all = pd.concat(all_curves, ignore_index=True)

df_pivot = df_all.pivot(index="RunDate", columns="Instrument", values="TRDPRC_1")

print(df_pivot)