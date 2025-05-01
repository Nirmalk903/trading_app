import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import json


symbols = ['^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS','AXISBANK.NS','BAJFINANCE.NS']

def getdata_vbt(symbol, period='20y', interval='1d'):
    
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join('./Underlying_data_vbt', file_name)
        
        Price = vbt.YFData.download(symbol, period=period, interval=interval).get("Close")
        High = vbt.YFData.download(symbol, period=period, interval=interval).get("High")
        Low = vbt.YFData.download(symbol, period=period, interval=interval).get("Low")
        Open = vbt.YFData.download(symbol, period=period, interval=interval).get("Open")
        Volume = vbt.YFData.download(symbol, period=period, interval=interval).get("Volume")
        
        data = pd.concat([Price, High, Low, Open, Volume], axis=1)
        data.columns = ['Price', 'High', 'Low', 'Open', 'Volume']
        
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        data.to_csv(file_path, index=False)
        print(f"Data for {symbol} saved successfully.")
    
    return None

getdata_vbt(symbols)