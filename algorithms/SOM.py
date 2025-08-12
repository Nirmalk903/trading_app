import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files
from tqdm import tqdm
import shap

import time

symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
print(f"Symbols selected: {symbols}")

def load_features(symbols, data_dir="Engineered_data"):
    import pandas as pd
    from datetime import datetime, timedelta

    dfs = []
    # Calculate the cutoff date for the last 5 years
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=5)
    print(f"Loading features for last 5 years for {len(symbols)} symbols...")
    for symbol in tqdm(symbols, desc="Loading features"):
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            # Ensure 'Date' is datetime
            if not np.issubdtype(df['Date'].dtype, np.datetime64):
                df['Date'] = pd.to_datetime(df['Date'])
            # Filter for last 5 years
            df = df[df['Date'] >= cutoff_date]
            dfs.append(df)
            print(f"  Loaded {symbol}: {len(df)} rows")
        else:
            print(f"  File not found for {symbol}")
    print("Feature loading complete.\n")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

start_time = time.time()
print("Starting SOM analysis pipeline...\n")
d = load_features(symbols)
print(f"Loaded DataFrame shape: {d.shape}")

print("Checking for missing values in features...")
print(d.isna().sum())

def select_uncorrelated_features(df, feature_cols, n_clusters=5):
    print("Selecting uncorrelated features using KMeans clustering on correlation matrix...")
    corr = df[feature_cols].corr().abs()
    dist = 1 - corr
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(dist)
    labels = kmeans.labels_
    selected_features = []
    for cluster in range(n_clusters):
        cluster_features = [feature_cols[i] for i in range(len(feature_cols)) if labels[i] == cluster]
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
        else:
            avg_corr = corr.loc[cluster_features, cluster_features].mean().sort_values()
            selected_features.append(avg_corr.index[0])
    print(f"Selected features: {selected_features}\n")
    return selected_features

def run_som_analysis(symbols, plot=True, save_dir="SOM_Image"):
    print("Running SOM analysis...")
    df = load_features(symbols)
    feature_cols = [col for col in df.columns if col not in ['Date', 'symbol','Close','High','Low','Volume'] and np.issubdtype(df[col].dtype, np.number)]
    print(f"Feature columns considered: {feature_cols}")
    df = df.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    print(f"Data shape after dropping rows with all NaNs in features: {df.shape}")
    if len(feature_cols) > 1:
        selected_features = select_uncorrelated_features(df, feature_cols, n_clusters=min(5, len(feature_cols)))
    else:
        selected_features = feature_cols
    print(f"Selected features for SOM: {selected_features}")
    X = df[selected_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaled for SOM.")

    som = MiniSom(3, 3, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    print("Training SOM...")
    som.train_random(X_scaled, 1000)
    print("SOM training complete.")

    df['SOM_cluster'] = [som.winner(x) for x in X_scaled]
    print("Assigned SOM clusters to each row.")

    # Ensure save directory exists
    save_path = os.path.join(os.path.dirname(__file__), save_dir)
    os.makedirs(save_path, exist_ok=True)

    # Plot SOM chart (showing all points, colored by symbol)
    if plot:
        print("Saving SOM clusters plot...")
        plt.figure(figsize=(7, 7))
        plt.title("SOM Clusters (all historical data)")
        symbols_unique = df['symbol'].unique()
        colors = plt.colormaps['tab20']
        for i, symbol in enumerate(symbols_unique):
            idx = df['symbol'] == symbol
            color = colors(i % colors.N)
            for x in X_scaled[idx]:
                w = som.winner(x)
                plt.plot(w[0]+0.1, w[1]+0.1, 'o', markersize=6, color=color, alpha=0.5)
            # Optionally, label the last point for each symbol
            last_idx = df[df['symbol'] == symbol].index[-1]
            w_last = som.winner(X_scaled[last_idx])
            plt.text(w_last[0]+0.1, w_last[1]+0.1, symbol, fontsize=10, color=color)
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.grid(True)
        plt.xlabel("SOM X")
        plt.ylabel("SOM Y")
        som_img_path = os.path.join(save_path, "som_clusters.png")
        plt.savefig(som_img_path, bbox_inches='tight')
        plt.close()
        print(f"SOM plot saved to {som_img_path}.")

    print("Calculating cluster centers (feature means per cluster)...")
    cluster_centers = {}
    for cluster in set(df['SOM_cluster']):
        idx = df['SOM_cluster'] == cluster
        cluster_centers[cluster] = df.loc[idx, selected_features].mean()
    print("Cluster centers calculated.\n")
    return cluster_centers

symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
cluster_centers = run_som_analysis(symbols, plot=True, save_dir="SOM_Image")

# Print all cluster centers
print("\nAll SOM cluster centers:")
for cluster, center in cluster_centers.items():
    print(f"\nCluster {cluster}:")
    print(center)

# Example: Find clusters with low RSI (potentially oversold)
print("\nChecking for clusters with low RSI (potentially oversold):")
for cluster, center in cluster_centers.items():
    if center.get('RSI', 50) < 35:
        print(f"\nCluster {cluster} may be oversold:")
        print(center)

# Convert cluster_centers to DataFrame for easier analysis
import pandas as pd
centers_df = pd.DataFrame(cluster_centers).T
print("\nCluster centers DataFrame:")
print(centers_df)

# --- Add: Table of SOM clusters and associated stocks ---
print("\nTable of SOM clusters and associated stocks:")
# Reload the last df with SOM_cluster assignments
def get_last_som_df(symbols):
    # This function should match the df used in run_som_analysis
    df = load_features(symbols)
    feature_cols = [col for col in df.columns if col not in ['Date', 'symbol','Close','High','Low','Volume'] and np.issubdtype(df[col].dtype, np.number)]
    df = df.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    if len(feature_cols) > 1:
        selected_features = select_uncorrelated_features(df, feature_cols, n_clusters=min(5, len(feature_cols)))
    else:
        selected_features = feature_cols
    X = df[selected_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    som = MiniSom(3, 3, X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, 1000)
    df['SOM_cluster'] = [som.winner(x) for x in X_scaled]
    return df

som_df = get_last_som_df(symbols)
cluster_stock_table = som_df.groupby('SOM_cluster')['symbol'].unique().reset_index()
cluster_stock_table['symbol'] = cluster_stock_table['symbol'].apply(lambda x: ', '.join(sorted(set(x))))
print(cluster_stock_table.to_string(index=False))

# Optionally, save the table to CSV
table_path = os.path.join(os.path.dirname(__file__), "SOM_Image", "cluster_stock_table.csv")
cluster_stock_table.to_csv(table_path, index=False)
print(f"\nCluster-stock association table saved to {table_path}.")

# Plot heatmap of cluster centers
import seaborn as sns
import matplotlib.pyplot as plt

print("Saving heatmap of cluster centers...")
plt.figure(figsize=(10, 6))
sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("SOM Cluster Centers (Feature Means)")
heatmap_img_path = os.path.join(os.path.dirname(__file__), "SOM_Image", "som_cluster_centers_heatmap.png")
plt.savefig(heatmap_img_path, bbox_inches='tight')
plt.close()
print(f"Heatmap saved to {heatmap_img_path}.")

end_time = time.time()
print(f"\nSOM analysis pipeline completed in {int(end_time - start_time)} seconds.")