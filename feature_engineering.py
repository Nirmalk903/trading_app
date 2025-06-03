
import pandas as pd
import numpy as np
# import pandas_ta as ta
import os
import vectorbt as vbt
from volatility_modeling import garch_vol
from scipy.stats import percentileofscore
# import talib as ta



class FeatureEngineering:
    
    def __init__(self,X):
        
        self.X = X
        self.High = X.High
        self.Low = X.Low
        self.Close = X.Close
        self.Open = X.Open
        self.Volume = X.Volume
        
    def __call__(self, X):
        self.callme(X)
        
    def dCPR(self,X): #creates daily central pivot range
        
        X['dCPR'] = (np.array(X[['High','Low','Close']].shift(1))).mean(axis=1)

        # Calculting Daily PCR to Low, High and Close

        # X['Low2dCPR'] = X['Low']/ X['dCPR']
        # X['High2dCPR'] = X['High']/ X['dCPR']
        X['Close2dCPR'] = X['Close']/ X['dCPR']
        
        return X
    
    
    def wCPR(self, X):  # Creates weekly pivot range
    # Ensure the index is a DatetimeIndex
        if not isinstance(X.index, pd.DatetimeIndex):
            X.set_index('Date', inplace=True)

        self.logic = {'High': 'max', 'Low': 'min', 'Close': 'last'}
        df1 = X.sort_index()
        df1 = df1.resample('W').apply(self.logic)
        df1.index = df1.index - pd.tseries.frequencies.to_offset("6D")
        df1.rename(columns={'High': 'wHigh', 'Low': 'wLow', 'Close': 'wClose'}, inplace=True)

        df1['wCPR'] = (np.array(df1[['wHigh', 'wLow', 'wClose']].shift(1))).mean(axis=1)

        for i in range(len(X.index)):
            for j in range(len(df1.index)):
                if (X.index[i] >= df1.index[j]) & (X.index[i] <= df1.index[j] + pd.offsets.Day(6)):
                    X.loc[X.index[i], 'wCPR'] = df1.loc[df1.index[j], 'wCPR']

        # X['Low2wCPR'] = X['Low'] / X['wCPR']
        # X['High2wCPR'] = X['High'] / X['wCPR']
        X['Close2wCPR'] = X['Close'] / X['wCPR']

        return X
    
    def cpr_vol(self,X,lag): #creates volatility columns for daily and weekly CPRs
        
        self.lag = lag
        vol_lag = range(5,lag,5)

        col = X.filter(regex='2d|2wCPR')

        for i in vol_lag:
            for j in col:
                X[f"{j}_{i}dVol"] = np.array(X[j].rolling(i).std()*np.sqrt(251))
        return X
    
    def price_momentum(self,X,lag):
        self.lag=lag
        for i in range(1,lag):
            X[f'Price_Moment_{i}'] = X.Close-X.Close.shift(i)
            
        return X
    
    def sma(self, X, lag):
        self.lag=lag
        for i in range(10,lag,5):
            
            X[f'SMA_{i}'] = X.Close.rolling(i).mean()
        
        return X
    
    def price_range(self,X):
        
        # Calculating intra-day price range

        X['O-C']= X['Open']-X['Close']
        X['H-L'] = X['High'] - X['Low']
        
        return X
    
    def dpo_centered(self, X, period = 20):
        
        if 'Close' not in X.columns:
            raise ValueError("The DataFrame must contain a 'Close' column.")

        # Calculate the Simple Moving Average (SMA)
        sma = X['Close'].rolling(window=period).mean()

        # Shift the SMA by (period / 2) + 1 periods
        sma_shifted = sma.shift(int((period / 2) + 1))

        # Calculate the DPO
        X[f'DPO_{period}'] = X['Close'] - sma_shifted
            
        return X
    
    
    def cfo(self, X, period = 20):
        self.period = period
        X['CFO'] = ta.cfo(X['Close'], length=self.period, fillna=True)
        X['CFO'] = X['CFO'].replace([np.inf, -np.inf], np.nan)
        X['CFO'] = X['CFO'].fillna(method='bfill')
        return X
    
    def rolling_vol(self, X, lag):
        self.lag=lag
        for i in range(5,lag,5):
            
            X[f'Rolling_vol_{i}'] = np.log(X.Close).diff().rolling(i).std()*np.sqrt(251)*100
        
        return X
    
    def returns(self,X):
        X['Returns'] = X['Close'].pct_change()
        
        return X
    
    def rsi(self,X):
        rsi = vbt.RSI.run(X['Close'], window=14, short_name='RSI')
        X['RSI'] = rsi.rsi
        return X
    
    def bollinger_bands(self,X, period=20, std=2):
        # Calculate the rolling mean and standard deviation
        rolling_mean = X['Close'].rolling(window=period).mean()
        rolling_std = X['Close'].rolling(window=period).std()

        # Calculate the upper and lower Bollinger Bands
        X['Bollinger_High'] = rolling_mean + (rolling_std * std)
        X['Bollinger_Low'] = rolling_mean - (rolling_std * std)

        return X
    
    def keltner_channel(self,X, period=20,atr_multiplier=1.5):
        # Calculate the rolling mean and average true range (ATR)
        X['Previous_Close'] = X['Close'].shift(1)
        X['True_Range'] = np.maximum(X['High'] - X['Low'], 
                                      np.maximum(abs(X['High'] - X['Previous_Close']), 
                                                 abs(X['Low'] - X['Previous_Close'])))
        X['ATR'] = X['True_Range'].rolling(window=period).mean()
        X.drop(columns=['Previous_Close', 'True_Range'], inplace=True)
        rolling_mean = X['Close'].rolling(window=period).mean()
        # Calculate the upper and lower Keltner Channels
        X['Keltner_High'] = rolling_mean + (X['ATR'] * atr_multiplier)
        X['Keltner_Low'] = rolling_mean - (X['ATR'] * atr_multiplier)

        return X
    
    def identify_volatility_squeezes(self, X, period=20):
        # Calculate the Bollinger Bands
        rolling_mean = X['Close'].rolling(window=period).mean()
        rolling_std = X['Close'].rolling(window=period).std()
        X['Bollinger_High'] = rolling_mean + (rolling_std * 2)
        X['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

        # Calculate the Keltner Channels
        X['Previous_Close'] = X['Close'].shift(1)
        X['True_Range'] = np.maximum(X['High'] - X['Low'], 
                                      np.maximum(abs(X['High'] - X['Previous_Close']), 
                                                 abs(X['Low'] - X['Previous_Close'])))
        X['ATR'] = X['True_Range'].rolling(window=period).mean()
        rolling_mean_keltner = X['Close'].rolling(window=period).mean()
        X.drop(columns=['Previous_Close', 'True_Range'], inplace=True)
        X['Keltner_High'] = rolling_mean_keltner + (X['ATR'] * 1.5)
        X['Keltner_Low'] = rolling_mean_keltner - (X['ATR'] * 1.5)

        # Identify volatility squeezes
        X['Volatility_Squeeze'] = ((X['Bollinger_High'] - X['Bollinger_Low']) < 
                                    (X['Keltner_High'] - X['Keltner_Low']))
        
        return X


    
    def callme(self,X):
        self.returns(X)
        self.dCPR(X)
        self.wCPR(X)
        # self.cpr_vol(X,lag=20)
        self.price_momentum(X,lag=20)
        self.sma(X,lag=20)
        self.price_range(X)
        self.dpo_centered(X)
        # self.dpo_non_centered(X,lag=20)
        # self.cfo(X,lag=20)
        self.rolling_vol(X,lag=25)
        self.rsi(X)
        self.bollinger_bands(X, period=20, std=2)
        self.keltner_channel(X, period=20, atr_multiplier=1.5)

        
        print('Feature Engineering Completed')
        return X



def add_features(symbols, interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        # yf_symbol = '^NSEI' if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'
        
        # Load the data
        data_path = f'./Underlying_data_vbt/{symbol}_{interval}.csv'
        data_path = os.path.join(data_path)
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} not found. Please download the data first.")
            continue
        else:
            data = pd.read_csv(data_path, parse_dates=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        
        # Apply feature engineering using callme
        feat_eng = FeatureEngineering(data)
        data = feat_eng.callme(data)
        
        # Add GARCH volatility
        data['garch_vol'] = garch_vol(symbol).values
        data['garch_vol'] = data['garch_vol'].bfill()
        data['garch_vol_pct'] = data['garch_vol'].diff()
        # Calculate percentiles
        data['garch_vol_percentile'] = [np.round(percentileofscore(data['garch_vol'], i),0) for i in data['garch_vol'] ]
        data['garch_vol_percentile'] = data['garch_vol_percentile'].bfill()
    
        
        # Save the engineered features to a new CSV file
        new_dir = f'./Engineered_data'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}_features.csv'
        file_path = os.path.join(new_dir, file_name)
        # data.to_csv(file_path, index=True)
        data.to_json(file_path.replace('.csv', '.json'), orient='records', lines=True)
        print(f"Feature engineered data for {symbol} saved successfully.")
    
    return None




def update_features(symbols, interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        
        # Load the raw data
        data_path = f'./Underlying_data_vbt/{symbol}_{interval}.csv'
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} not found. Please download the data first.")
            continue

        raw_data = pd.read_csv(data_path)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d', errors='coerce')
        raw_data.set_index('Date', inplace=True)

        # Check if processed data already exists
        processed_path = f'./Engineered_data/{symbol}_{interval}_features.json'
        if os.path.exists(processed_path):
            processed_data = pd.read_json(processed_path, orient='records', lines=True)
            processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%Y-%m-%d', errors='coerce')
            processed_data.set_index('Date', inplace=True)

            # Find new data that needs to be processed
            new_data = raw_data.loc[~raw_data.index.isin(processed_data.index)]
            if new_data.empty:
                print(f"No new data to process for {symbol}.")
                continue

            print(f"Processing {len(new_data)} new rows for {symbol}.")
        else:
            print(f"No existing processed data found for {symbol}. Processing all data.")
            new_data = raw_data

        # Apply feature engineering to the new data
        feat_eng = FeatureEngineering(new_data)
        new_data = feat_eng.callme(new_data)

        # Add GARCH volatility
        new_data['garch_vol'] = garch_vol(symbol).values[-len(new_data):]
        new_data['garch_vol'] = new_data['garch_vol'].bfill()
        new_data['garch_vol_pct_chng'] = new_data['garch_vol'].pct_change()
        new_data['garch_vol_percentile'] = [
            np.round(percentileofscore(raw_data['garch_vol'], i), 0) for i in raw_data['garch_vol']
        ]

        # Combine the new data with the existing processed data
        if os.path.exists(processed_path):
            combined_data = pd.concat([processed_data, new_data])
        else:
            combined_data = new_data

        # Save the updated data to a JSON file
        combined_data.reset_index(inplace=True)
        combined_data.to_json(processed_path, orient='records', lines=True)
        print(f"Feature engineered data for {symbol} saved successfully.")

    return combined_data
