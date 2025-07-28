import streamlit as st
import pandas as pd
# import matplotlib as plt
from PIL import Image
from feature_engineering import add_features
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
import os
import time
from datetime import datetime as dt_time

st.title("Trading Analytics Dashboard")

st.write("Use the controls above to select date and filter symbols, then click 'Run Analytics'.")

# Select date and top N symbols
dates = get_dates_from_most_active_files()
selected_date = st.selectbox("Select Date", dates[::-1])
top_n = st.slider("Number of Top Symbols", 1, 20, 10)

all_symbols = get_symbols(selected_date, top_n=top_n)[0]

# # Dropdown for single symbol selection
# selected_symbol = st.selectbox(
#     "Select a single symbol for analysis", options=all_symbols
# )

# # Add filters: multi-select for symbols
selected_symbols = st.multiselect(
    "Filter and Select Symbols", options=all_symbols, default=all_symbols
)

st.write(f"Selected symbols: {selected_symbols}")

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
    dt = dt_time.now().strftime('%Y-%m-%d')
    st.subheader(f"GARCH Vol Percentile vs RSI  {dt}")
    df1 = plot_garch_vs_rsi(selected_symbols)
    # st.write(df1)
    # if df1 is not None:
    #     st.dataframe(df1)
    #     fig, ax = plt.subplots()
    #     ax.scatter(df1['RSI'], df1['garch_vol_percentile'])
    #     for _, row in df1.iterrows():
    #         ax.text(row['RSI'], row['garch_vol_percentile'], row['symbol'], fontsize=9)
    #     st.pyplot(fig)

    # Load and display image from Images folder
    
    image_path = os.path.join("Images", f"garch_vs_rsi_{dt}.png")
    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption="GARCH vs RSI Scatter Plot", use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

    # # Plot GARCH vs Avg IV
    # st.subheader("GARCH Vol Percentile vs Avg IV")
    # fig2 = plot_garch_vs_avg_iv(selected_symbols)
    # if fig2 is not None:
    #     st.pyplot(fig2)



# To run: streamlit run app.py


