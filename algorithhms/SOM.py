import pandas as pd
import numpy as np
from minisom import MiniSom
import os

def load_feature_data(symbol, data_dir="Engineered_data"):
    """Load feature engineered data for a symbol."""
    file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
    if not os.path.exists(file_path):
        print(f"[{symbol}] Feature file not found.")
        return None
    print(f"[{symbol}] Loading feature data...")
    df = pd.read_json(file_path, orient='records', lines=True)
    if "Date" not in df.columns:
        print(f"[{symbol}] 'Date' column not found. Skipping.")
        return None
    df = df.sort_values("Date")
    return df

def train_som_on_features(df, som_shape=(5, 5), seed=42, symbol=""):
    """Train SOM on all numeric features (excluding Date, Close, and non-numeric columns) and return cluster assignments."""
    # Exclude non-feature columns
    exclude_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'symbol']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    print(f"[{symbol}] Training SOM on features: {feature_cols}")
    X = df[feature_cols].dropna().values
    som = MiniSom(som_shape[0], som_shape[1], len(feature_cols), sigma=1.0, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, 1000)
    print(f"[{symbol}] SOM training complete.")
    # Assign each row to a SOM cluster (winning node)
    win_map = np.array([som.winner(x) for x in X])
    # Flatten cluster index for easier use
    cluster_labels = [f"{i}_{j}" for i, j in win_map]
    df = df.loc[df[feature_cols].dropna().index].copy()
    df['SOM_Cluster'] = cluster_labels
    return df, som

def generate_signals_from_clusters(df, cluster_col='SOM_Cluster', price_col='Close', symbol=""):
    """Generate simple trading signals based on SOM clusters' average future returns."""
    print(f"[{symbol}] Generating trading signals from SOM clusters...")
    df['fwd_return'] = df[price_col].shift(-5) / df[price_col] - 1
    cluster_stats = df.groupby(cluster_col)['fwd_return'].mean()
    cluster_signal = cluster_stats.apply(lambda x: 'buy' if x > 0 else 'sell')
    df['signal'] = df[cluster_col].map(cluster_signal)
    print(f"[{symbol}] Signal generation complete.")
    return df

def backtest_signals(df, price_col='Close', symbol=""):
    """Simple backtest: buy on 'buy' signal, sell (exit) on 'sell' signal, no shorting."""
    print(f"[{symbol}] Running backtest on generated signals...")
    df = df.copy()
    df['position'] = (df['signal'] == 'buy').astype(int)
    df['returns'] = df[price_col].pct_change().fillna(0)
    df['strategy_returns'] = df['returns'] * df['position'].shift(1).fillna(0)
    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    print(f"[{symbol}] Backtest complete. Cumulative return: {cumulative_return:.2%}")
    return cumulative_return, df

def get_all_symbols(data_dir="Engineered_data"):
    """Get all symbols from feature files in the directory."""
    files = [f for f in os.listdir(data_dir) if f.endswith("_1d_features.json")]
    symbols = [f.split("_1d_features.json")[0] for f in files]
    return symbols

if __name__ == "__main__":
    data_dir = "Engineered_data"
    symbols = get_all_symbols(data_dir)
    results = []

    print(f"Found {len(symbols)} symbols to process.\n")

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] Processing symbol: {symbol}")
        df = load_feature_data(symbol, data_dir)
        if df is None or df.empty:
            print(f"[{symbol}] Skipping due to missing or insufficient data.")
            continue
        df_som, som = train_som_on_features(df, symbol=symbol)
        df_signals = generate_signals_from_clusters(df_som, symbol=symbol)
        cum_return, df_bt = backtest_signals(df_signals, symbol=symbol)
        results.append({'symbol': symbol, 'cumulative_return': cum_return})

    results_df = pd.DataFrame(results).sort_values('cumulative_return', ascending=False)
    print("\nMost profitable trading strategies by symbol:")
    print(results_df)

    if not results_df.empty:
        best_symbol = results_df.iloc[0]['symbol']
        print(f"\nMost profitable symbol: {best_symbol}")