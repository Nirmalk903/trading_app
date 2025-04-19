import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_random, retry
from functools import lru_cache
from datetime import datetime, timedelta
from Options_Utility import atm_strike, time_to_expiry
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
from json import JSONDecodeError
import json
import time
import os

# Function to fetch options data from NSE website with retry logic

@lru_cache()
@retry(wait=wait_random(min=0.1, max=1))
def fetch_options_data(symbol):
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY','BANKNIFTY','MIDCPNIFTY','FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
                'accept-encoding': 'gzip, deflate, br, zstd',
                'accept-language': 'en-US,en;q=0.9',
                'accept': '*/*'}
    session = requests.Session()
    request = session.get(url, headers=headers)
    print(request.status_code)
    response = session.get(url, headers=headers, cookies=dict(request.cookies))
    
    return pd.DataFrame(response.json())

# Alternative function to fetch options data from NSE Website

@lru_cache()
def fetch_live_options_data(symbol):
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"
    
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'en-US,en;q=0.9',
        'accept': '*/*'
    }
                
    session = requests.Session()
    while True:
        request = session.get(url, headers=headers)
        if request.status_code == 200:
            cookies = dict(request.cookies)
            break
        else:
            time.sleep(0.5)  # Wait for 0.5 seconds before retrying

    while True:
        response = session.get(url, headers=headers, cookies=cookies)
        if response.status_code == 200:
            print(f"Success! Status code 200 received. {symbol} saved")
            break
        else:
            time.sleep(0.5)

    df = response.json()
    # df.to_csv(f'{symbol}_options_data.csv', index=False)
    return pd.DataFrame(df)


def fetch_and_save_options_chain(symbol):
    symbol = symbol.upper()
    print(f'printing option chain for {symbol}')
    # data = fetch_options_data(symbol)
    data = fetch_live_options_data(symbol)
    dates = pd.to_datetime(data.loc['expiryDates','records'])
    max_expiry = dates[0]+timedelta(days=90)
    expiry = [i.strftime('%d-%b-%Y') for i in dates if dates[0] <= i <= max_expiry]
    
    ls = []
    for dt in expiry:
        rawoptions = pd.DataFrame(data['records']['data']).fillna(0)
        # rawoptions = rawoptions.query('expiryDates == @dt').reset_index(drop=True)
        rawoptions = rawoptions[rawoptions['expiryDate']==dt].reset_index(drop=True)
        
        for i in range(0,len(rawoptions)):
            calloi=callcoi=cltp=putoi=putcoi=pltp=iv=0
            stp = rawoptions['strikePrice'][i]
            if(rawoptions['CE'][i]==0):
                calloi=callcoi=0
            else:
                calloi=rawoptions['CE'][i]['openInterest']
                callcoi=rawoptions['CE'][i]['changeinOpenInterest']
                cltp=rawoptions['CE'][i]['lastPrice']
                civ=rawoptions['CE'][i]['impliedVolatility']
            
            if(rawoptions['PE'][i]==0):
                putoi=putcoi=0
            else:
                putoi=rawoptions['PE'][i]['openInterest']
                putcoi=rawoptions['PE'][i]['changeinOpenInterest']
                pltp=rawoptions['PE'][i]['lastPrice']
                piv=rawoptions['PE'][i]['impliedVolatility']
                expiry = rawoptions['PE'][i]['expiryDate']
            
            optdata = {'Expiry':expiry,'call_oi':calloi,'call_change_oi':callcoi, 'call_ltp':cltp, 'strike_price':stp,
                   'strike_price':stp,'put_ltp':pltp, 'put_oi':putoi,'put_change_oi':putcoi}
    
            ls.append(optdata)
        OptionChain = pd.DataFrame(ls)
        # del ls
        new_dir = f'./OptionChainJSON'
        os.makedirs(new_dir, exist_ok=True)
        file_path = os.path.join(new_dir, f'{symbol}_OptionChain.json')
        OptionChain.to_json(file_path,orient='records')
    return f'Option Chain Saved'

fetch_and_save_options_chain('reliance')
# def enrich_option_chain(symbol):
#     symbol = symbol.upper()
#     print(f'Enriching option chain for {symbol}')
    