
# The CUSUM filter is a sequential analysis technique used for monitoring change detection. It is particularly useful in time series analysis for identifying shifts in the mean level of a process.  

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files

def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos = max(0, sPos + diff.loc[i])
        sNeg = min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def plot_cusum_events_all(symbols, engineered_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "cusum_events_close_charts.pdf")
    with PdfPages(pdf_path) as pdf:
        for symbol in symbols:
            file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
            if not os.path.exists(file_path):
                continue
            df = pd.read_json(file_path, orient='records', lines=True)
            if "Date" not in df.columns or "Close" not in df.columns:
                continue
            df = df.sort_values("Date")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            h = df["Close"].std() * 0.2
            t_events = getTEvents(df["Close"], h)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["Close"], label="Close Price")
            plt.scatter(t_events, df.loc[t_events, "Close"], color='red', label="CUSUM Events", zorder=5)
            plt.title(f"{symbol} - Close Price with CUSUM Events")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"All Close price charts have been saved to {pdf_path}")

def plot_garch_cusum_events_all(symbols, engineered_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "cusum_events_garch_vol_charts.pdf")
    with PdfPages(pdf_path) as pdf:
        for symbol in symbols:
            file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
            if not os.path.exists(file_path):
                continue
            df = pd.read_json(file_path, orient='records', lines=True)
            if "Date" not in df.columns or "garch_vol" not in df.columns:
                continue
            df = df.sort_values("Date")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            h = df["garch_vol"].std() * 0.25
            t_events = getTEvents(df["garch_vol"], h)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["garch_vol"], label="GARCH Volatility")
            plt.scatter(t_events, df.loc[t_events, "garch_vol"], color='red', label="CUSUM Events", zorder=5)
            plt.title(f"{symbol} - GARCH Volatility with CUSUM Events")
            plt.xlabel("Date")
            plt.ylabel("GARCH Volatility")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"All GARCH Volatility charts have been saved to {pdf_path}")

# --- Main execution ---
engineered_dir = "./Engineered_data"
results_dir = "./results"
symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=25)[0]

plot_cusum_events_all(symbols, engineered_dir, results_dir)
# plot_garch_cusum_events_all(symbols, engineered_dir, results_dir)