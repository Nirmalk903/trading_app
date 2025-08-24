import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files

def tick_rule(prices):
    """Implements the tick rule for a price series."""
    price_diff = prices.diff()
    b = []
    prev_b = 1
    for diff in price_diff:
        if pd.isna(diff):
            b.append(prev_b)
        elif diff == 0:
            b.append(prev_b)
        else:
            val = np.sign(diff)
            b.append(val)
            prev_b = val
    return pd.Series(b, index=prices.index)

def expected_theta(T_hist, b_hist, span=20):
    """Eo[θ] = Eo[T] * (2*P[b=1] - 1) using EWMA."""
    if len(T_hist) == 0 or len(b_hist) == 0:
        return 50  # arbitrary start threshold
    E_T = pd.Series(T_hist).ewm(span=span).mean().iloc[-1]
    P_b1 = pd.Series(b_hist).eq(1).ewm(span=span).mean().iloc[-1]
    return E_T * (2 * P_b1 - 1)

def tick_imbalance_bars(prices, expma_span=20):
    """Returns a list of bar end indices for tick imbalance bars."""
    b = tick_rule(prices)
    bars = []
    T_hist = []
    b_hist = []
    t0 = 0
    while t0 < len(prices):
        theta = 0
        t = t0
        while t < len(prices):
            theta += b.iloc[t]
            if len(T_hist) >= 2:
                E_theta = expected_theta(T_hist, b_hist, span=expma_span)
            else:
                E_theta = 50  # arbitrary start threshold
            if abs(theta) >= abs(E_theta):
                bars.append(t)
                T_hist.append(t - t0 + 1)
                b_hist.extend(b.iloc[t0:t+1].tolist())
                t0 = t + 1
                break
            t += 1
        else:
            break
    return bars

def plot_tick_imbalance_bars_all_symbols(engineered_dir, results_dir, symbols):
    os.makedirs(results_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "tick_imbalance_bars.pdf")
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
            bars = tick_imbalance_bars(df["Close"])
            plt.figure(figsize=(14, 6))
            plt.plot(df.index, df["Close"], label="Close Price")
            for bar_idx in bars:
                plt.axvline(df.index[bar_idx], color='red', linestyle='--', alpha=0.5)
            plt.title(f"{symbol} - Tick Imbalance Bars")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"All tick imbalance bar plots saved to {pdf_path}")

if __name__ == "__main__":
    engineered_dir = "./Engineered_data"
    results_dir = "./results"
    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
    plot_tick_imbalance_bars_all_symbols(engineered_dir, results_dir, symbols)