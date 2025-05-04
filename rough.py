import pickle
import os

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
    
# Note: The loaded model can be used for further analysis or predictions as needed.
# The above code assumes that the EGARCH model was saved using the pickle module in the previous code snippet.
# Ensure that the EGARCH model is compatible with the loaded data and libraries.
