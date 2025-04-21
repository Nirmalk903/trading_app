import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_random, retry
from functools import lru_cache
from datetime import datetime, timedelta
from Options_Utility import atm_strike, time_to_expiry
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
from json import JSONDecodeError
import json
import time
import os
from get_data import fetch_live_options_data, fetch_options_data, fetch_and_save_options_chain



symbols = ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY', 'RELIANCE', 'TATAMOTORS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS','AXISBANK','BAJFINANCE']

for symbol in symbols:
    try:
        # Fetch and save options chain data
        fetch_and_save_options_chain(symbol)
        
        # Print the fetched data
        print(f"Live data for {symbol}:")
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")