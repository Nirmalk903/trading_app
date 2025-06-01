import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


  
def plot_garch_vs_rsi(symbols, data_dir='./Engineered_data'):
    """
    Reads feature files for each symbol and creates a scatter plot of GARCH Vol Percentile vs RSI.
    Highlights NIFTY and BANKNIFTY markers in different colors.
    """
    records = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if not os.path.exists(file_path):
            print(f"File not found for {symbol}: {file_path}")
            continue
        df = pd.read_json(file_path, orient='records', lines=True)
        if 'garch_vol_percentile' not in df.columns or 'RSI' not in df.columns:
            print(f"Required columns not found in {file_path}")
            continue
        last_row = df.iloc[-1]
        records.append({
            'symbol': symbol,
            'garch_vol_percentile': int(last_row['garch_vol_percentile']),
            'RSI': int(last_row['RSI'])
        })
    df1 = pd.DataFrame(records)

    plt.figure(figsize=(8, 6))
    # Plot all except NIFTY and BANKNIFTY
    mask_nifty = df1['symbol'].str.upper().isin(['NIFTY', '^NSEI'])
    mask_banknifty = df1['symbol'].str.upper().isin(['BANKNIFTY', '^NSEBANK'])
    mask_others = ~(mask_nifty | mask_banknifty)

    plt.scatter(df1.loc[mask_others, 'RSI'], df1.loc[mask_others, 'garch_vol_percentile'],
                alpha=0.5, label='Others', color='blue')
    plt.scatter(df1.loc[mask_nifty, 'RSI'], df1.loc[mask_nifty, 'garch_vol_percentile'],
                alpha=0.9, label='NIFTY', color='red', marker='o', s=80, edgecolor='black')
    plt.scatter(df1.loc[mask_banknifty, 'RSI'], df1.loc[mask_banknifty, 'garch_vol_percentile'],
                alpha=0.9, label='BANKNIFTY', color='green', marker='s', s=80, edgecolor='black')

    # Add symbol labels to each marker
    for _, row in df1.iterrows():
        plt.text(row['RSI'], row['garch_vol_percentile'], row['symbol'], fontsize=9, ha='right', va='bottom')

    plt.ylabel('GARCH Vol Percentile')
    plt.xlabel('RSI')
    plt.title('GARCH Vol Percentile vs RSI')
    plt.legend()
    plt.grid(True)
    plt.savefig('garch_vs_rsi_scatter_plot.png')
    plt.close()
    return None
