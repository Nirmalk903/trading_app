import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from tqdm import tqdm
from hmmlearn.hmm import GaussianHMM

def load_engineered_data(symbols, data_dir="Engineered_data"):
    dfs = []
    for symbol in tqdm(symbols, desc="Loading features"):
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            if "Date" not in df.columns or df.empty:
                continue
            if not np.issubdtype(df['Date'].dtype, np.datetime64):
                df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def apply_hidden_markov_model(df, n_states=2, target_col='Returns'):
    """
    Applies a Gaussian Hidden Markov Model to the target column.
    Returns a DataFrame with hidden states and state probabilities.
    """
    results = []
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].sort_values('Date').copy()
        y = sdf[target_col].dropna().values.reshape(-1, 1)
        if len(y) < 100:
            continue  # Not enough data for stable estimation
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
            model.fit(y)
            hidden_states = model.predict(y)
            sdf = sdf.iloc[-len(y):].copy()
            sdf['hmm_state'] = hidden_states
            # Posterior probabilities for each state
            posteriors = model.predict_proba(y)
            for i in range(n_states):
                sdf[f'state_{i}_prob'] = posteriors[:, i]
            results.append(sdf)
            print(f"{symbol}: HMM fitted. Score={model.score(y):.2f}")
        except Exception as e:
            print(f"{symbol}: HMM failed - {e}")
    return pd.concat(results) if results else pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    from data_download_vbt import get_symbols, get_dates_from_most_active_files

    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=10)[0]
    df = load_engineered_data(symbols)
    if not df.empty and 'Returns' in df.columns:
        df_hmm = apply_hidden_markov_model(df, n_states=2, target_col='Returns')
        if not df_hmm.empty:
            # Save results
            out_path = os.path.join(os.path.dirname(__file__), "hidden_markov_model_results.csv")
            df_hmm.to_csv(out_path, index=False)
            print(f"\nHMM results saved to {out_path}")
        else:
            print("No HMM results generated.")
    else:
        print("No data loaded or 'Returns' column missing.")

    # Read the file from the algorithms folder
    out_path = os.path.join(os.path.dirname(__file__), "hidden_markov_model_results.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print("\nHMM State Statistics by Stock:")
        for symbol in df['symbol'].unique():
            print(f"\nSymbol: {symbol}")
            print(df[df['symbol'] == symbol].groupby('hmm_state')['Returns'].agg(['mean', 'std', 'count']))
    else:
        print(f"{out_path} does not exist.")

