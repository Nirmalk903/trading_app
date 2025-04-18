
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_random
from functools import lru_cache
from datetime import datetime, timedelta


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
    response = session.get(url, headers=headers, cookies=dict(request.cookies))
    
    return pd.DataFrame(response.json())



def fetch_and_save_options_chain(symbol):
    symbol = symbol.upper()
    print(f'printing option chain for {symbol}')
    data = fetch_options_data(symbol)
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
           
        # OptionChain.to_csv(f'{symbol}_OptionChain.csv',index=False)
        OptionChain.to_json(f'{symbol}_OptionChain.json',orient='records')
    return f'Option Chain Saved'

def enrich_option_chain(symbol):
    symbol = symbol.upper()
    print(f'Enriching option chain for {symbol}')
    