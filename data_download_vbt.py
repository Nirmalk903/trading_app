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
        
        Close = vbt.YFData.download(symbol, period=period, interval=interval).get("Close")
        High = vbt.YFData.download(symbol, period=period, interval=interval).get("High")
        Low = vbt.YFData.download(symbol, period=period, interval=interval).get("Low")
        Open = vbt.YFData.download(symbol, period=period, interval=interval).get("Open")
        Volume = vbt.YFData.download(symbol, period=period, interval=interval).get("Volume")
        
        data = pd.concat([Close, High, Low, Open, Volume], axis=1)
        data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        data.to_csv(file_path, index=False)
        print(f"Data for {symbol} saved successfully.")
    
    return None



symbols = ['^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS','AXISBANK.NS','BAJFINANCE.NS']

def get_underlying_data_vbt(symbols, period='20y', interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join(new_dir, file_name)

        # Check if file exists and get last date
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, parse_dates=['Date'])
            if not existing_data.empty:
                last_date = existing_data['Date'].max()
                # Download only data after last_date
                start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                Close = vbt.YFData.download(symbol, start=start, interval=interval).get("Close")
                High = vbt.YFData.download(symbol, start=start, interval=interval).get("High")
                Low = vbt.YFData.download(symbol, start=start, interval=interval).get("Low")
                Open = vbt.YFData.download(symbol, start=start, interval=interval).get("Open")
                Volume = vbt.YFData.download(symbol, start=start, interval=interval).get("Volume")
                new_data = pd.concat([Close, High, Low, Open, Volume], axis=1)
                new_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
                new_data.reset_index(inplace=True)
                new_data['Date'] = pd.to_datetime(new_data['Date'], format='%Y-%m-%d', errors='coerce')
                new_data['Date'] = new_data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                # Only append if there is new data
                if not new_data.empty:
                    combined = pd.concat([existing_data, new_data], ignore_index=True)
                    combined.drop_duplicates(subset='Date', keep='last', inplace=True)
                    combined.to_csv(file_path, index=False)
                    print(f"Appended {len(new_data)} new rows for {symbol}.")
                else:
                    print(f"No new data for {symbol}.")
            else:
                # If file exists but is empty, download all data
                Close = vbt.YFData.download(symbol, period=period, interval=interval).get("Close")
                High = vbt.YFData.download(symbol, period=period, interval=interval).get("High")
                Low = vbt.YFData.download(symbol, period=period, interval=interval).get("Low")
                Open = vbt.YFData.download(symbol, period=period, interval=interval).get("Open")
                Volume = vbt.YFData.download(symbol, period=period, interval=interval).get("Volume")
                data = pd.concat([Close, High, Low, Open, Volume], axis=1)
                data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
                data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                data.to_csv(file_path, index=False)
                print(f"Data for {symbol} saved successfully.")
        else:
            # If file does not exist, download all data
            Close = vbt.YFData.download(symbol, period=period, interval=interval).get("Close")
            High = vbt.YFData.download(symbol, period=period, interval=interval).get("High")
            Low = vbt.YFData.download(symbol, period=period, interval=interval).get("Low")
            Open = vbt.YFData.download(symbol, period=period, interval=interval).get("Open")
            Volume = vbt.YFData.download(symbol, period=period, interval=interval).get("Volume")
            data = pd.concat([Close, High, Low, Open, Volume], axis=1)
            data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
            data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            data.to_csv(file_path, index=False)
            print(f"Data for {symbol} saved successfully.")

    return None
