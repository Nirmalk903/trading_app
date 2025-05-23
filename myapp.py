
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except Exception:
    pass



import pandas as pd
import numpy as np
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
import os
from get_options_data import fetch_and_save_options_chain, enrich_option_chain
from data_download_vbt import getdata_vbt, get_underlying_data_vbt
from feature_engineering import add_features


dt = datetime.now()

def get_symbols(dt,top_n=17):
    dt = dt.strftime('%d-%b-%Y')
    nifty_fity_path = os.path.join('./Nifty_Fifty', "MW-NIFTY-50.csv")
    if os.path.exists(nifty_fity_path):
        nifty_fifty = pd.read_csv(nifty_fity_path)
        nifty_symbols = nifty_fifty['SYMBOL \n'].tolist()
        liquid_symbols = nifty_symbols + ['NIFTY', 'BANKNIFTY']
    else:
        print(f"File {nifty_fity_path} does not exist.")
    
    file_name = f'LA-MOST-ACTIVE-UNDERLYING-{dt}.csv'
    file_path = os.path.join('./Most_Active_Underlying', file_name)
    if os.path.exists(file_path):
        most_active = pd.read_csv(file_path)
        most_active = most_active[~most_active['Symbol'].isin(['MIDCPNIFTY', 'FINNIFTY','NIFTYIT','NIFTYNXT50','NIFTYPSUBANK','NIFTYINFRA','NIFTYMETAL','NIFTYPHARMA','NIFTYMEDIA','NIFTYAUTO','NIFTYCONSUMPTION','NIFTYENERGY','NIFTYFMCG','NIFTYHEALTHCARE'])].reset_index(drop=True)
        most_active = most_active[most_active['Symbol'].isin(liquid_symbols)].reset_index(drop=True)
        most_active['YF_Symbol'] = most_active['Symbol'].apply(lambda x: '^NSEI' if x == 'NIFTY' else '^NSEBANK' if x == 'BANKNIFTY' else f'{x}.NS')
        most_active = most_active.sort_values(by='Value (? Lakhs) - Total', ascending=False).reset_index(drop=True)
        most_active = most_active.head(top_n)
    else:
        print(f"File {file_name} does not exist.")
        
    symbols = most_active['Symbol'].tolist()
    yf_symbols = most_active['YF_Symbol'].tolist()
    return symbols, yf_symbols

symbols , yf_symbols = get_symbols(dt,top_n=17)

for symbol in symbols:
    try:
        # Fetch and save options chain data
        fetch_and_save_options_chain(symbol) #Download data from NSE, formats it into an option chain and save it to a JSON file
        enrich_option_chain(symbol)      #Enrich the data with greeks and other calculations and save it to a JSON file
        
        # Print the fetched data
        print(f"Live data for {symbol} saved successfully.")
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        

def load_enriched_option_chain(symbol):        
    symbol = symbol.upper()
    print(f'Loading option chain for {symbol}')
    file_name = f'{symbol}_OptionChain_Enriched.json'
    file_path = os.path.join('./OptionChainJSON_Enriched', file_name)
    chain = pd.read_json(file_path, orient='records')
    return chain


def load_atm_chain(symbol):
    symbol = symbol.upper()
    print(f'Loading ATM data for {symbol}')
    file_name = f'{symbol}_ATM_OptionChain.json'
    file_path = os.path.join('./ATM_OptionChainJSON', file_name)
    chain = pd.read_json(file_path, orient='records')
    print(f"ATM data for {symbol} loaded successfully.")
    return chain

# Create pipeline for underlying data and option chain data

# Downloading underlying data and adding features 
# getdata_vbt(yf_symbol, period='10y', interval='1d')
get_underlying_data_vbt(yf_symbols, period='10y', interval='1d')  # Download underlying data
add_features(yf_symbols)  # Add features to the underlying data
# Feature Engineering

add_features(['SBIN.NS'])


yf_symbols

# Load the enriched option chain data


