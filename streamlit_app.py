import streamlit as st
import pandas as pd
# from market_calendar import stock_earnings_calendar
from PIL import Image
from feature_engineering import add_features
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
import os
import time
from datetime import datetime as dt_time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import plotly.graph_objects as go
import numpy as np


st.title("Trading Analytics Dashboard")

st.write("Use the controls above to select date and filter symbols, then click 'Run Analytics'.")

# Select date and top N symbols
dates = get_dates_from_most_active_files()
selected_date = st.selectbox("Select Date", dates[::-1])
top_n = st.slider("Number of Top Symbols", 1, 20, 10)

all_symbols = get_symbols(selected_date, top_n=top_n)[0]

# # Add filters: multi-select for symbols
selected_symbols = st.multiselect(
    "Filter and Select Symbols", options=all_symbols, default=all_symbols
)

# st.write(f"Selected symbols: {selected_symbols}")

# Caching functions
def was_run_recently(symbol, cache_dir="./analytics_cache", max_age=3600):
    """Return True if analytics for symbol was run in last max_age seconds."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{symbol}_last_run.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            last_run = float(f.read().strip())
        if time.time() - last_run < max_age:
            return True
    return False

def mark_run(symbol, cache_dir="./analytics_cache"):
    cache_file = os.path.join(cache_dir, f"{symbol}_last_run.txt")
    with open(cache_file, "w") as f:
        f.write(str(time.time()))

if st.button("Run Analytics"):
    progress_text = "Running analytics. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    total = len(selected_symbols)
    skipped = []
    for i, symbol in enumerate(selected_symbols):
        if was_run_recently(symbol):
            skipped.append(symbol)
            my_bar.progress((i + 1) / total, text=f"Skipping {symbol} (recently processed)")
            continue
        get_underlying_data_vbt([symbol], period='10y', interval='1d')
        add_features([symbol])
        mark_run(symbol)
        my_bar.progress((i + 1) / total, text=f"Processing {symbol} ({i+1}/{total})")
    st.success("Features added for selected symbols.")
    if skipped:
        st.info(f"Skipped (already processed in last hour): {', '.join(skipped)}")

# Plot GARCH vs RSI
# st.header("GARCH Volatility Percentile vs RSI")
dt = dt_time.now().strftime('%Y-%m-%d')
st.subheader(f"GARCH Vol Percentile vs RSI  {dt}")
df1 = plot_garch_vs_rsi(selected_symbols)

image_path = os.path.join("Images", f"garch_vs_rsi_{dt}.png")
if os.path.exists(image_path):
    st.image(Image.open(image_path), caption="GARCH vs RSI Scatter Plot", use_container_width=True)
else:
    st.warning(f"Image not found: {image_path}")

# Section: Plot Historical Chart from Engineered_data

st.header("Plot Historical Chart")

# Dropdown to select a single stock for historical chart
hist_symbol = st.selectbox(
    "Select stock for historical chart", options=all_symbols, key="hist_symbol"
)

# Select number of days to show (period)
period_options = {
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "All": None
}
selected_period_label = st.selectbox(
    "Select period", list(period_options.keys()), index=2
)
num_days = period_options[selected_period_label]

if st.button("Show Historical Chart"):
    feature_file = os.path.join("Engineered_data", f"{hist_symbol}_1d_features.json")
    if os.path.exists(feature_file):
        df_hist = pd.read_json(feature_file, orient='records', lines=True)
        df_hist = df_hist.sort_values("Date")
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        # Calculate moving averages
        df_hist["MA_10"] = df_hist["Close"].rolling(window=10).mean()
        df_hist["MA_50"] = df_hist["Close"].rolling(window=50).mean()
        df_hist["MA_100"] = df_hist["Close"].rolling(window=100).mean()
        
        df_hist_recent = df_hist.tail(num_days) if num_days is not None else df_hist

        # Check for minimum data length
        min_rows = 100
        if len(df_hist_recent) < min_rows:
            st.warning(f"Not enough data to plot all indicators (need at least {min_rows} rows, got {len(df_hist_recent)}).")
        else:
            df_mpf = df_hist_recent.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

            apds = []
            # Only add plots if the data is not all-NaN
            if not np.all(np.isnan(df_hist_recent["wCPR"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["wCPR"].values, panel=0,
                                             type='scatter', markersize=0.5, color='blue', marker='o', ylabel='wCPR'))
            if not np.all(np.isnan(df_hist_recent["MA_10"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["MA_10"].values, panel=0,
                                             type='line', color='green', width=1.2, ylabel='MA 10'))
            if not np.all(np.isnan(df_hist_recent["MA_50"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["MA_50"].values, panel=0,
                                             type='line', color='orange', width=1.2, ylabel='MA 50'))
            if not np.all(np.isnan(df_hist_recent["MA_100"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["MA_100"].values, panel=0,
                                             type='line', color='purple', width=1.2, ylabel='MA 100'))
            if not np.all(np.isnan(df_hist_recent["RSI"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["RSI"].values, panel=1,
                                             type='line', color='grey', width=1.2, ylabel='RSI'))
            if not np.all(np.isnan(df_hist_recent["garch_vol"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["garch_vol"].values, panel=2,
                                             type='bar', color='red', width=1.2, ylabel='Volatility'))
            if not np.all(np.isnan(df_hist_recent["garch_vol_percentile"].values)):
                apds.append(mpf.make_addplot(df_hist_recent["garch_vol_percentile"].values, panel=3,
                                             type='line', color='orange', width=1.2, ylabel='VolP'))

            # Add squared returns to panel 4 (5th panel, index 4) using 'Returns' column from engineered data
            if "Returns" in df_hist_recent.columns:
                squared_returns = df_hist_recent["Returns"] ** 2
                if not np.all(np.isnan(squared_returns.values)):
                    apds.append(mpf.make_addplot(
                        squared_returns.values,
                        panel=4,
                        type='line',
                        color='brown',
                        width=1.2,
                        ylabel='Squared Ret'
                    ))

            panel_ratios = (6, 1, 1, 1, 1)  # Add extra panel for squared returns

            fig, axlist = mpf.plot(
                df_mpf,
                type='candle',
                style='yahoo',
                addplot=apds,
                panel_ratios=panel_ratios,
                returnfig=True,
                figsize=(10, 10)
            )

            # Add symbol name to center top of OHLC panel
            axlist[0].text(
                0.5, 0.98, f"{hist_symbol}",
                transform=axlist[0].transAxes,
                fontsize=16,
                fontweight='bold',
                va='top',
                ha='center',
                color='navy',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            )

            # Add custom legend for moving averages
            ma_lines = []
            ma_labels = []
            if not np.all(np.isnan(df_hist_recent["MA_10"].values)):
                ma_lines.append(axlist[0].plot([], [], color='green', linewidth=2)[0])
                ma_labels.append('MA 10')
            if not np.all(np.isnan(df_hist_recent["MA_50"].values)):
                ma_lines.append(axlist[0].plot([], [], color='orange', linewidth=2)[0])
                ma_labels.append('MA 50')
            if not np.all(np.isnan(df_hist_recent["MA_100"].values)):
                ma_lines.append(axlist[0].plot([], [], color='purple', linewidth=2)[0])
                ma_labels.append('MA 100')

            if ma_lines:
                axlist[0].legend(ma_lines, ma_labels, loc='upper left')

            st.pyplot(fig, use_container_width=True)
    else:
        st.warning(f"Feature file not found for {hist_symbol}: {feature_file}")

# Upcoming Earnings, Dividends & Corporate Actions


