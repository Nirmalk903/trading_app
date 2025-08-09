import os
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files

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

    # Calculate percentiles for adaptive thresholds
    rsi_high = np.percentile(latest['RSI'].dropna(), 70)
    rsi_low = np.percentile(latest['RSI'].dropna(), 30)
    vol_high = np.percentile(latest['vol_percentile'].dropna(), 70)
    vol_low = np.percentile(latest['vol_percentile'].dropna(), 30)
    ma_high = np.percentile(latest['dist_from_MA20'].dropna(), 70)
    ma_low = np.percentile(latest['dist_from_MA20'].dropna(), 30)

    # Identify rich/cheap/neutral clusters based on technicals (adaptive)
    cluster_labels = {}
    for cluster, center in cluster_centers.items():
        if (
            center.get('RSI', 50) > rsi_high or
            center.get('vol_percentile', 50) > vol_high or
            center.get('dist_from_MA20', 0) > ma_high
        ):
            cluster_labels[cluster] = 'Rich/Overbought'
        elif (
            center.get('RSI', 50) < rsi_low or
            center.get('vol_percentile', 50) < vol_low or
            center.get('dist_from_MA20', 0) < ma_low
        ):
            cluster_labels[cluster] = 'Cheap/Oversold'
        else:
            cluster_labels[cluster] = 'Neutral/Trendless'

    # Generate notes
    notes = []
    for _, row in latest.iterrows():
        cluster = row['SOM_cluster']
        status = cluster_labels[cluster]
        comment = f"{row['symbol']} is classified as **{status}**."
        if status == 'Rich/Overbought':
            comment += (
                " Short-term technicals show: high RSI (momentum/overbought), high volatility percentile (active market), "
                "and price extended above the 20-day moving average. This may indicate a potential reversal, breakout, or increased risk for option sellers."
            )
        elif status == 'Cheap/Oversold':
            comment += (
                " Short-term technicals show: low RSI (weak momentum/oversold), low volatility percentile (quiet market), "
                "and price extended below the 20-day moving average. This may indicate a potential bounce, breakdown, or opportunity for option buyers."
            )
        else:
            comment += (
                " Technicals are neutral: RSI and volatility are moderate, and price is near the 20-day moving average. "
                "No strong short-term trend or edge detected."
            )
        notes.append(comment)
    return notes, cluster_labels

symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
notes, cluster_labels = run_som_analysis(symbols, plot=True)

print("\nCluster Labels:")
for cluster, label in cluster_labels.items():
    print(f"Cluster {cluster}: {label}")

print("\nNotes:")
for note in notes:
    print(f" - {note}")