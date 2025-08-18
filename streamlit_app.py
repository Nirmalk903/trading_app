import streamlit as st
import pandas as pd
# from market_calendar import stock_earnings_calendar
from PIL import Image
from feature_engineering import process_symbol, create_underlying_analytics
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

st.title("Trading Analytics Dashboard")

st.write("Use the controls below to select date and filter symbols, then click 'Run Analytics'.")

# Select date and top N symbols
dates = get_dates_from_most_active_files()
# Convert to string (date only, no time)
dates = [str(pd.to_datetime(d).date()) for d in dates]
selected_date = st.selectbox("Select Date", dates[::-1])
top_n = st.slider("Number of Top Symbols", 1, 30, 10)

# Convert selected_date string back to datetime for get_symbols
selected_date_dt = pd.to_datetime(selected_date)

all_symbols = get_symbols(selected_date_dt, top_n=top_n)[0]

# # Add filters: multi-select for symbols
st.markdown(
    """
    <style>
    .symbol-font .stMultiSelect label, .symbol-font .stMultiSelect span {
        font-size: 0.3em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
selected_symbols = st.multiselect(
    "Filter and Select Symbols", options=all_symbols, default=all_symbols, key="symbol_multiselect", help=None
)

# --- Feature Engineering Step (parallelized and cached) ---

if st.button("Run Analytics"):
    st.info("Running feature engineering for selected symbols. Please wait...")
    progress_bar = st.progress(0, text="Starting...")

    start_time = time.time()  # Start timer

    with st.spinner("Processing features..."):
        # Remove parallel execution, use create_underlying_analytics for all selected symbols
        create_underlying_analytics(selected_symbols)
        progress_bar.progress(1.0, text=f"Processed {len(selected_symbols)}/{len(selected_symbols)} symbols")

    elapsed = time.time() - start_time  # End timer
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    st.success(f"Feature engineering completed for selected symbols! Time taken: {minutes} min {seconds} sec.")
    progress_bar.empty()

# --- Load all engineered data for selected symbols only once ---
engineered_data = {}
for symbol in selected_symbols:
    feature_file = os.path.join("Engineered_data", f"{symbol}_1d_features.json")
    if os.path.exists(feature_file):
        df = pd.read_json(feature_file, orient='records', lines=True)
        if "Date" not in df.columns:
            if df.index.name == "Date":
                df = df.reset_index()
        if "Date" not in df.columns or df.empty:
            continue
        df = df.sort_values("Date")
        df["Date"] = pd.to_datetime(df["Date"])
        engineered_data[symbol] = df

# --- Use engineered_data for all calculations below ---

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

# Plot Historical Chart
st.header("Plot Historical Chart")
hist_symbol = st.selectbox(
    "Select stock for historical chart", options=selected_symbols, key="hist_symbol"
)

# Select number of days to show (period)
period_options = {
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "All": None
}
selected_period_label = st.selectbox(
    "Select period", list(period_options.keys()), index=2
)
num_days = period_options[selected_period_label]

if st.button("Show Historical Chart"):
    if hist_symbol in engineered_data:
        df_hist = engineered_data[hist_symbol].copy()
        # Calculate moving averages
        df_hist["MA_10"] = df_hist["Close"].rolling(window=10).mean()
        df_hist["MA_50"] = df_hist["Close"].rolling(window=50).mean()
        df_hist["MA_100"] = df_hist["Close"].rolling(window=100).mean()
        
        df_hist_recent = df_hist.tail(num_days) if num_days is not None else df_hist

        # --- Add summary table for latest row ---
        latest = df_hist_recent.iloc[-1]
        summary_dict = {
            "Latest Price": latest["Close"],
            "Daily Return": latest.get("Returns", None),
            "GARCH Volatility": latest.get("garch_vol", None),
            "GARCH Volatility Percentile": latest.get("garch_vol_percentile", None),
            "Daily CPR": latest.get("dCPR", None),
            "RSI": latest.get("RSI", None),
            "RSI Percentile": latest.get("RSI_percentile", None),
            "Weekly RSI": latest.get("RSI_weekly", None),
            "Weekly RSI Percentile": latest.get("RSI_percentile_weekly", None)
        }
        # Format numeric values: 'Daily Return' as percentage with one decimal, others as 0 decimals
        for k, v in summary_dict.items():
            if k == "Daily Return" and isinstance(v, (int, float)) and v is not None:
                summary_dict[k] = f"{v*100:.1f}%"
            elif isinstance(v, (int, float)) and v is not None:
                summary_dict[k] = f"{v:.0f}"
        summary_df = pd.DataFrame([summary_dict])

        # --- Display as tabular format with heading ---
        st.subheader(f"Stock Analysis - {latest['Date'].date()}")
        st.table(summary_df)

        # --- Existing plotting code ---
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
        st.warning(f"Feature file not found for {hist_symbol}")

# --- Summary table for all symbols (latest row for each) ---
summary_rows = []
for symbol, df in engineered_data.items():
    if df.empty or "Date" not in df.columns:
        continue
    latest = df.iloc[-1]
    summary_rows.append({
        "Symbol": symbol,
        "Date": latest["Date"].date(),
        "Latest Price": latest["Close"],
        "Daily Return": latest.get("Returns", None),
        "GARCH Volatility": latest.get("garch_vol", None),
        "GARCH Volatility Percentile": latest.get("garch_vol_percentile", None),
        "Vol_Change": latest.get("garch_vol_pct", None),
        "Daily CPR": latest.get("dCPR", None),
        "RSI": latest.get("RSI", None),
        "RSI Percentile": latest.get("RSI_percentile", None),
        "Weekly RSI": latest.get("RSI_weekly", None),
        "Weekly RSI Percentile": latest.get("RSI_percentile_weekly", None)
    })

if summary_rows:
    summary_all_df = pd.DataFrame(summary_rows)
    # Remove the 'Date' column and get the latest date for the header
    latest_date = summary_all_df["Date"].iloc[0] if "Date" in summary_all_df.columns else ""
    summary_all_df = summary_all_df.drop(columns=["Date"])

    # Keep a copy of the original numeric columns for sorting
    numeric_cols = ["Latest Price", "Daily Return", "GARCH Volatility", "GARCH Volatility Percentile",
                    "Daily CPR", "RSI", "RSI Percentile", "Weekly RSI", "Weekly RSI Percentile"]

    # Convert numeric columns to float for sorting
    for col in numeric_cols:
        if col in summary_all_df.columns:
            summary_all_df[col] = pd.to_numeric(summary_all_df[col], errors='coerce')

    # Let user select column to sort by
    sort_col = st.selectbox(
        "Sort table by column",
        options=["Symbol"] + [col for col in numeric_cols if col in summary_all_df.columns],
        index=2  # Default to "Daily Return"
    )
    ascending = st.radio("Sort order", ["Descending", "Ascending"], index=0) == "Ascending"
    summary_all_df = summary_all_df.sort_values(by=sort_col, ascending=ascending)

    # Format columns for display
    for col in summary_all_df.columns:
        if col == "Daily Return":
            summary_all_df[col] = summary_all_df[col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notnull(x) else ""
            )
        elif col in numeric_cols and col != "Daily Return":
            summary_all_df[col] = summary_all_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "")

    # Reset index to start from 1
    summary_all_df.index = summary_all_df.index + 1

    # --- Highlight Daily Return: green for positive, red for negative, center align ---
    def highlight_daily_return(val):
        try:
            num = float(val.replace('%', ''))
            color = 'green' if num > 0 else 'red' if num < 0 else 'black'
            return f'color: {color}; text-align: center;'
        except:
            return 'text-align: center;'

    styled_df = summary_all_df.style.map(highlight_daily_return, subset=['Daily Return']) \
                                    .set_properties(**{'text-align': 'center'})

    st.subheader(f"Stock Analysis - {latest_date}")
    st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("No feature files found for the selected symbols.")


# Upcoming Earnings, Dividends & Corporate Actions

# --- Correlation Matrix for Selected Stocks ---
st.header("Correlation Matrix for Selected Stocks")

close_prices = {}
for symbol, df in engineered_data.items():
    if not df.empty and "Date" in df.columns:
        close_prices[symbol] = df.set_index("Date")["Close"]

if close_prices:
    close_df = pd.DataFrame(close_prices)
    returns_df = close_df.pct_change(fill_method=None)
    corr_matrix = returns_df.corr()
    st.subheader("Correlation Matrix (Daily Returns)")
    st.dataframe(
        corr_matrix.style.format("{:.2f}").background_gradient(cmap='coolwarm'),
        use_container_width=True
    )
else:
    st.warning("Not enough data to compute correlation matrix for selected stocks.")


