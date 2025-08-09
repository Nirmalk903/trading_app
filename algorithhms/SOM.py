import os
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files

def load_features(symbols, data_dir="Engineered_data"):
    dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def select_uncorrelated_features(df, feature_cols, n_clusters=5):
    # Compute correlation matrix
    corr = df[feature_cols].corr().abs()
    # Convert correlation to distance
    dist = 1 - corr
    # KMeans expects a 2D array, so flatten the distance matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(dist)
    labels = kmeans.labels_
    selected_features = []
    for cluster in range(n_clusters):
        cluster_features = [feature_cols[i] for i in range(len(feature_cols)) if labels[i] == cluster]
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
        else:
            # Select the feature with lowest average correlation to others in the cluster
            avg_corr = corr.loc[cluster_features, cluster_features].mean().sort_values()
            selected_features.append(avg_corr.index[0])
    return selected_features

def run_som_analysis(symbols, plot=True):
    df = load_features(symbols)
    feature_cols = [col for col in df.columns if col not in ['Date', 'symbol','Close','High','Low','Volume'] and np.issubdtype(df[col].dtype, np.number)]
    # Use latest row for each symbol
    latest = df.sort_values('Date').groupby('symbol').tail(1).reset_index(drop=True)
    # Select uncorrelated features using KMeans
    if len(feature_cols) > 1:
        selected_features = select_uncorrelated_features(latest, feature_cols, n_clusters=min(5, len(feature_cols)))
    else:
        selected_features = feature_cols
    X = latest[selected_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    som = MiniSom(3, 3, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, 500)

    # Assign clusters
    latest['SOM_cluster'] = [som.winner(x) for x in X_scaled]

    # Plot SOM chart
    if plot:
        plt.figure(figsize=(6, 6))
        plt.title("SOM Clusters (each node may have multiple stocks)")
        for i, x in enumerate(X_scaled):
            w = som.winner(x)
            plt.text(w[0]+0.1, w[1]+0.1, latest.iloc[i]['symbol'], fontsize=12)
            plt.plot(w[0]+0.1, w[1]+0.1, 'o', markersize=10)
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.grid(True)
        plt.xlabel("SOM X")
        plt.ylabel("SOM Y")
        plt.show()

    # Analyze clusters
    cluster_centers = {}
    for cluster in set(latest['SOM_cluster']):
        idx = latest['SOM_cluster'] == cluster
        cluster_centers[cluster] = latest.loc[idx, selected_features].mean()

    # Identify rich/cheap clusters (example: based on 'Close' and 'Volatility')
    cluster_richness = {}
    for cluster, center in cluster_centers.items():
        if 'Close' in latest.columns and center.get('Close', 0) > np.percentile(latest['Close'], 75):
            cluster_richness[cluster] = 'Expensive'
        elif 'Close' in latest.columns and center.get('Close', 0) < np.percentile(latest['Close'], 25):
            cluster_richness[cluster] = 'Cheap'
        else:
            cluster_richness[cluster] = 'Fair'

    # Generate notes
    notes = []
    for _, row in latest.iterrows():
        cluster = row['SOM_cluster']
        status = cluster_richness[cluster]
        volatility = row.get('garch_vol', np.nan)
        comment = f"{row['symbol']} is classified as **{status}**."
        if status == 'Expensive':
            comment += " The stock is trading at a higher price compared to peers."
        elif status == 'Cheap':
            comment += " The stock is trading at a lower price compared to peers."
        else:
            comment += " The stock is fairly priced compared to peers."
        if not np.isnan(volatility):
            if volatility > np.percentile(latest['garch_vol'], 75):
                comment += " Volatility is high, expect larger price swings."
            elif volatility < np.percentile(latest['garch_vol'], 25):
                comment += " Volatility is low, price is relatively stable."
            else:
                comment += " Volatility is moderate."
        notes.append(comment)
    return notes

# Example usage:
if __name__ == "__main__":
    symbols = get_symbols(get_dates_from_most_active_files()[-1],top_n=17)[0]
    # symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY','AXISBANK']  # Replace with your symbols
    notes = run_som_analysis(symbols, plot=True)
    for note in notes:
        print(note)