import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import time
import re




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
        # print(most_active.head())
        most_active = most_active[~most_active['Symbol'].isin(['MIDCPNIFTY', 'FINNIFTY','NIFTYIT','NIFTYNXT50','NIFTYPSUBANK','NIFTYINFRA','NIFTYMETAL','NIFTYPHARMA','NIFTYMEDIA','NIFTYAUTO','NIFTYCONSUMPTION','NIFTYENERGY','NIFTYFMCG','NIFTYHEALTHCARE'])].reset_index(drop=True)
        most_active = most_active[most_active['Symbol'].isin(liquid_symbols)].reset_index(drop=True)
        most_active['YF_Symbol'] = most_active['Symbol'].apply(lambda x: '^NSEI' if x == 'NIFTY' else '^NSEBANK' if x == 'BANKNIFTY' else f'{x}.NS')
        most_active = most_active.sort_values(by='Value (₹ Lakhs) - Total', ascending=False).reset_index(drop=True)
        most_active = most_active.head(top_n)
    else:
        print(f"File {file_name} does not exist.")
        
    symbols = most_active['Symbol'].tolist()
    yf_symbols = most_active['YF_Symbol'].tolist()
    return symbols, yf_symbols





def getdata_vbt(symbols, period='20y', interval='1d'):
    
    yf_symbol = ['^NSEI' if symbol == 'NIFTY' else '^NSEBANK' if symbol == 'BANKNIFTY' else f'{symbol}.NS' for symbol in symbols]
    
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        yf_symbol = "^NSEI" if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'
        
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join('./Underlying_data_vbt', file_name)
        
        Close = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Close")
        High = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("High")
        Low = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Low")
        Open = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Open")
        Volume = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Volume")
        
        data = pd.concat([Close, High, Low, Open, Volume], axis=1)
        data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        data.to_csv(file_path, index=False)
        print(f"Data for {symbol} saved successfully.")
    
    return None


def get_underlying_data_vbt(symbols, period='20y', interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        yf_symbol = '^NSEI' if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'
        new_dir = f'./Underlying_data_vbt'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}.csv'
        file_path = os.path.join(new_dir, file_name)

        # Check if file exists and get last date
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, parse_dates=['Date'])
            if not existing_data.empty:
                last_date = existing_data['Date'].max()
                today = pd.to_datetime(datetime.now().date())
                if pd.to_datetime(last_date).date() == today.date():
                    start = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                else:
                    start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                # Download only data after last_date
                # start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                Close = vbt.YFData.download(yf_symbol, start=start, interval=interval).get("Close")
                High = vbt.YFData.download(yf_symbol, start=start, interval=interval).get("High")
                Low = vbt.YFData.download(yf_symbol, start=start, interval=interval).get("Low")
                Open = vbt.YFData.download(yf_symbol, start=start, interval=interval).get("Open")
                Volume = vbt.YFData.download(yf_symbol, start=start, interval=interval).get("Volume")
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
                Close = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Close")
                High = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("High")
                Low = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Low")
                Open = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Open")
                Volume = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Volume")
                data = pd.concat([Close, High, Low, Open, Volume], axis=1)
                data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
                data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                data.to_csv(file_path, index=False)
                print(f"Data for {symbol} saved successfully.")
        else:
            # If file does not exist, download all data
            Close = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Close")
            High = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("High")
            Low = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Low")
            Open = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Open")
            Volume = vbt.YFData.download(yf_symbol, period=period, interval=interval).get("Volume")
            data = pd.concat([Close, High, Low, Open, Volume], axis=1)
            data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
            data['Date'] = data['Date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            data.to_csv(file_path, index=False)
            print(f"Data for {symbol} saved successfully.")

    return None

# get_underlying_data_vbt(['ITC'], period='10y', interval='1d')



def get_dates_from_most_active_files(folder='./Most_Active_Underlying'):
    # Regex to extract date from filenames like LA-MOST-ACTIVE-UNDERLYING-23-May-2024.csv
    date_pattern = re.compile(r'LA-MOST-ACTIVE-UNDERLYING-(\d{2}-[A-Za-z]{3}-\d{4})\.csv')
    dates = []
    for fname in os.listdir(folder):
        match = date_pattern.match(fname)
        if match:
            dates.append(match.group(1))
    dates_sorted = pd.to_datetime(sorted(dates, key=lambda x: datetime.strptime(x, "%d-%b-%Y")))
    return dates_sorted
