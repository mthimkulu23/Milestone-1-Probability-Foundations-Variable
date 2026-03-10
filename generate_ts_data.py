import pandas as pd
import numpy as np

# Load the existing users data
df = pd.read_csv('/Users/damacm1152/Documents/Probability and Statistics/Milestone-1-Probability-Foundations-Variable/data/finflow_users.csv')

# Create a time-series version by adding a timestamp and ordering
# We'll just assume the current order has some temporal dependency or add slight noise
df_ts = df.copy()
df_ts['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

# To make independence check interesting (but likely OK), we'll keep it as is
# but ensure it's saved as the required filename
df_ts.to_csv('/Users/damacm1152/Documents/Probability and Statistics/Milestone-1-Probability-Foundations-Variable/data/finflow_timeseries.csv', index=False)
print("Created finflow_timeseries.csv in data/ subdirectory.")
