import pandas as pd
import numpy as np
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
import os
from get_data import fetch_and_save_options_chain, enrich_option_chain

symbols = ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY', 'RELIANCE', 'TATAMOTORS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS','AXISBANK','BAJFINANCE']

for symbol in symbols:
    try:
        # Fetch and save options chain data
        fetch_and_save_options_chain(symbol)
        enrich_option_chain(symbol)
        
        # Print the fetched data
        print(f"Live data for {symbol} saved successfully.")
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        

def load_option_chain(symbol):
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


enrich_option_chain('NIFTY')
df = load_atm_chain('NIFTY')
df
