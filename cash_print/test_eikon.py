import eikon as ek
ek.set_app_key("1e01b72982374e88971ea95fe42801910d7207ef")
df = ek.get_timeseries('PAAAL00', start_date='2026-01-01', end_date='2026-07-01')
print(df)