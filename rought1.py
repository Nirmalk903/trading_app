
import os
import re
import pandas as pd
from datetime import datetime
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols

def get_dates_from_most_active_files(folder='./Most_Active_Underlying'):
    # Regex to extract date from filenames like LA-MOST-ACTIVE-UNDERLYING-23-May-2024.csv
    date_pattern = re.compile(r'LA-MOST-ACTIVE-UNDERLYING-(\d{2}-[A-Za-z]{3}-\d{4})\.csv')
    dates = []
    for fname in os.listdir(folder):
        match = date_pattern.match(fname)
        if match:
            dates.append(match.group(1))
    dates_sorted = pd.to_datetime(sorted(dates, key=lambda x: datetime.strptime(x, "%d-%b-%Y")))
    return dates_sorted

# Example usage:
dates = get_dates_from_most_active_files()
dates


prev_symbols = get_symbols(dates[-2],top_n=17)[0]

curr_symbols = get_symbols(dates[-1],top_n=17)[0]

curr_symbols_set = set(curr_symbols)
prev_symbols_set = set(prev_symbols)
new_symbols = curr_symbols_set - prev_symbols_set
excluded_symbols = prev_symbols_set - curr_symbols_set
print(f"Symbols excluded in the New Most Active Underlying file: {excluded_symbols}")
print(f"New symbols in the latest Most Active Underlying file: {new_symbols}")