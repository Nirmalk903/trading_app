import pandas as pd
import numpy as np
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
import os
from tqdm import tqdm
from get_options_data import fetch_and_save_options_chain, enrich_option_chain,load_enriched_option_chain, load_atm_chain
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
from feature_engineering import add_features
import matplotlib.pyplot as plt
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
# from market_calendar import  stock_earnings_calendar
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


symbols = get_symbols(get_dates_from_most_active_files()[-1],top_n=17)[0]
prev_symbols = get_symbols(get_dates_from_most_active_files()[-2],top_n=17)[0]
new_symbols = set(symbols) - set(prev_symbols)
symbol_excluded = set(prev_symbols) - set(symbols)

# symbols = ['M&M']

for _ in new_symbols:
    print(f"New symbol added: {_}") 
    
for _ in symbol_excluded:
    print(f"Symbol excluded: {_}")



def fetch_and_enrich(symbol):
    try:
        start_time = time.time()
        fetch_and_save_options_chain(symbol)
        elapsed = time.time() - start_time
        timeout = 180  # seconds
        if elapsed > timeout:
            print(f"Skipping {symbol} option chain: took more than {timeout} seconds (actual: {elapsed:.2f}s).")
            return False  # Indicate skip
        enrich_option_chain(symbol)
        total_elapsed = time.time() - start_time
        print(f"Successfully fetched and enriched data for {symbol} in {total_elapsed:.2f} seconds.")
        return True  # Indicate success
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return

def create_underlying_analytics(symbols):
    get_underlying_data_vbt(symbols, period='10y', interval='1d')
    add_features(symbols)
    # plot_garch_vs_rsi(symbols)
    return "Analytics created successfully for all symbols."

def create_options_analytics(symbols, max_workers=7):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_enrich, symbol) for symbol in symbols]
        for future in as_completed(futures):
            pass  # All logging is handled in fetch_and_enrich

    return "Analytics created successfully for Option Chains."


# Usage:

if __name__ == "__main__":
    
    create_underlying_analytics(symbols)
    # stock_earnings_calendar(symbols)
    # create_options_analytics(symbols, max_workers=4)
    
    # for symbol in symbols:
    #     fetch_and_enrich(symbol)
    
    # create_options_analytics(symbols[9:], max_workers=len(symbols[9:]))


# fetch_and_enrich('SBIN')  # Example of fetching and enriching a specific symbol


# plot_garch_vs_rsi(symbols)
# create_underlying_analytics(['RELIANCE'])
