import pandas as pd
import numpy as np
import os
import pickle
from arch import arch_model

symbols = ['^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'AXISBANK.NS', 'BAJFINANCE.NS']

def egarch(symbols):
    """Load data for each symbol, fit an EGARCH model, and save the optimized model."""
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        
        # Load the data
        data_path = f'./Underlying_data_vbt/{symbol}_1d.csv'
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} does not exist.")
            continue
        
        data = pd.read_csv(data_path)
        
        # Ensure 'Close' column is present
        if 'Close' not in data.columns:
            print(f"'Close' column not found in {symbol} data.")
            continue
        
        # Fit the EGARCH model
        returns = np.log(data['Close']).diff().dropna() * 100  # Convert to percentage returns
        g1 = arch_model(returns, vol='EGARCH', p=1,o=0, q=1, dist='Normal', rescale=False)
        g1_fit = g1.fit(disp='off')
        
        # Save the model results to a text file
        model_results_path = f'./EGARCH_Results/{symbol}_EGARCH_results.txt'
        os.makedirs(os.path.dirname(model_results_path), exist_ok=True)
        with open(model_results_path, 'w') as f:
            f.write(str(g1_fit.summary()))
        
        # Save the optimized model using pickle
        model_pickle_path = f'./EGARCH_Models/{symbol}_EGARCH_model.pkl'
        os.makedirs(os.path.dirname(model_pickle_path), exist_ok=True)
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(g1_fit, f)
        print(f"Optimized EGARCH model for {symbol} saved successfully.")

# Run the EGARCH function
egarch(symbols)




def load_egarch_model(symbol):
   
    model_pickle_path = f'./EGARCH_Models/{symbol}_EGARCH_model.pkl'

    if not os.path.exists(model_pickle_path):
        print(f"Model file for {symbol} does not exist.")
        return None

    with open(model_pickle_path, 'rb') as f:
        garch_model = pickle.load(f)

    print(f"EGARCH model for {symbol} loaded successfully.")
    return garch_model

# Example usage
symbol = 'RELIANCE.NS'  # Replace with the desired symbol
loaded_model = load_egarch_model(symbol)

if loaded_model:
    print(loaded_model.summary())
    