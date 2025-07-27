import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_download_vbt import get_symbols, get_dates_from_most_active_files
from feature_engineering import add_features
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv

st.title("Trading Analytics Dashboard")

# Select date and top N symbols
dates = get_dates_from_most_active_files()
selected_date = st.selectbox("Select Date", dates[::-1])
top_n = st.slider("Number of Top Symbols", 1, 20, 10)

all_symbols = get_symbols(selected_date, top_n=top_n)[0]

# Add filters: multi-select for symbols
selected_symbols = st.multiselect(
    "Filter and Select Symbols", options=all_symbols, default=all_symbols
)

st.write(f"Selected symbols: {selected_symbols}")

if st.button("Run Analytics"):
    progress_text = "Running analytics. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    total = len(selected_symbols)
    for i, symbol in enumerate(selected_symbols):
        add_features([symbol])
        my_bar.progress((i + 1) / total, text=f"Processing {symbol} ({i+1}/{total})")
    st.success("Features added for selected symbols.")

    # Plot GARCH vs RSI
    st.subheader("GARCH Vol Percentile vs RSI")
    df1 = plot_garch_vs_rsi(selected_symbols)
    if df1 is not None:
        st.dataframe(df1)
        fig, ax = plt.subplots()
        ax.scatter(df1['RSI'], df1['garch_vol_percentile'])
        for _, row in df1.iterrows():
            ax.text(row['RSI'], row['garch_vol_percentile'], row['symbol'], fontsize=9)
        st.pyplot(fig)

    # Plot GARCH vs Avg IV
    st.subheader("GARCH Vol Percentile vs Avg IV")
    fig2 = plot_garch_vs_avg_iv(selected_symbols)
    if fig2 is not None:
        st.pyplot(fig2)

st.write("Use the controls above to select date and filter symbols, then click 'Run Analytics'.")

# To run: streamlit run app.py


