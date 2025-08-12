import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import time
import vectorbt as vbt

def load_features(symbols, data_dir="Engineered_data"):
    dfs = []
    print(f"Loading features for {len(symbols)} symbols...")
    for symbol in symbols:
        file_path = f"{data_dir}/{symbol}_1d_features.json"
        try:
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            dfs.append(df)
            print(f"  Loaded: {symbol} ({len(df)} rows)")
        except Exception as e:
            print(f"  Skipped: {symbol} (Error: {e})")
            continue
    print("Feature loading complete.\n")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def filter_last_n_years(df, years=8):
    if "Date" not in df.columns:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    return df[df["Date"] >= cutoff].copy()

def prepare_data(df, target_return=0.01, target_period=3):
    print("Preparing data and creating target variable...")
    df = df.sort_values(['symbol', 'Date'])
    df['target'] = df.groupby('symbol')['Returns'].shift(-target_period)
    df['target'] = (df['target'] > target_return).astype(int)
    feature_cols = [col for col in df.columns if col not in ['Date', 'symbol','Close','High','Low','Volume'] and np.issubdtype(df[col].dtype, np.number)]
    df = df.dropna(subset=feature_cols + ['target'])
    X = df[feature_cols]
    y = df['target']
    print(f"Prepared data with {len(X)} samples and {len(feature_cols)} features.\n")
    return X, y, df

def select_features_boruta(X, y):
    print("Selecting features using BorutaPy (this may take a while)...")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=30)
    # Use a sample for speed
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    y_sample = y.loc[X_sample.index]
    feat_selector.fit(X_sample.values, y_sample.values)
    selected_features = X.columns[feat_selector.support_].tolist()
    print("Selected features by BorutaPy:", selected_features, "\n")
    return selected_features


def train_naive_bayes(X, y):
    print("Training Naive Bayes classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Naive Bayes Test Accuracy: {score:.2f}\n")
    return model

def identify_opportunities(model, df, feature_cols):
    print("Identifying option trading opportunities...")
    latest = df.sort_values('Date').groupby('symbol').tail(1)
    # Ensure all selected features are present in latest
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = np.nan
    X_latest = latest[feature_cols]
    probs = model.predict_proba(X_latest)[:, 1]
    latest['Opportunity_Prob'] = probs
    opportunities = latest[latest['Opportunity_Prob'] > 0.7]
    print(f"Found {len(opportunities)} opportunities with probability > 0.7.\n")
    return opportunities[['Date', 'symbol', 'Opportunity_Prob']]

def backtest_signals(df, model, feature_cols, threshold=0.7):
    print("\nRunning backtest using vectorbt...")
    # Predict probabilities for all rows
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X_all = df[feature_cols]
    probs = model.predict_proba(X_all)[:, 1]
    df['Opportunity_Prob'] = probs
    # Generate signals: 1 if prob > threshold, else 0
    df['signal'] = (df['Opportunity_Prob'] > threshold).astype(int)
    # Backtest for each symbol
    results = {}
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values('Date')
        price = sdf['Close'].values
        entries = sdf['signal'].values.astype(bool)
        # Simple exit: next day (1-bar hold)
        exits = np.roll(entries, 1)
        exits[0] = False
        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            freq='1D',
            init_cash=100_000
        )
        results[symbol] = pf
        print(f"\nBacktest for {symbol}:")
        print(pf.stats())
        fig = pf.plot(title=f"Backtest for {symbol}")
        fig.show(renderer="browser")
    return results




# Add this function to your script
def scenario_analysis(df, target_periods, target_returns, symbols, years=5, threshold=0.7):
    print("\n--- Scenario Analysis ---")
    results = []
    for period in target_periods:
        for ret in target_returns:
            print(f"\nTarget Period: {period} days, Target Return: {ret*100:.2f}%")
            df_filtered = filter_last_n_years(df, years=years)
            X, y, df_prep = prepare_data(df_filtered, target_return=ret, target_period=period)
            if len(X) < 50:
                print("  Not enough data for this scenario, skipping.")
                continue
            selected_features = X.columns.tolist()
            # Capture accuracy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = GaussianNB()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"Naive Bayes Test Accuracy: {score:.2f}\n")
            opportunities = identify_opportunities(model, df_prep, feature_cols=selected_features)
            for symbol in symbols:
                symbol_opps = opportunities[opportunities['symbol'] == symbol]
                if not symbol_opps.empty:
                    prob = float(symbol_opps['Opportunity_Prob'].iloc[0])
                else:
                    prob = None
                results.append({
                    "Symbol": symbol,
                    "Target_Period": period,
                    "Target_Return": ret,
                    "Opportunity_Prob": prob,
                    "NaiveBayes_Accuracy": score
                })
            print("Opportunities for this scenario:")
            print(opportunities[['symbol', 'Date', 'Opportunity_Prob']].to_string(index=False))
    # Convert results to DataFrame for summary
    results_df = pd.DataFrame(results)
    print("\n--- Scenario Analysis Summary ---")
    print(
        results_df.pivot_table(
            index=['Symbol'],
            columns=['Target_Period', 'Target_Return'],
            values=['Opportunity_Prob', 'NaiveBayes_Accuracy']
        ).fillna('-')
    )
    return results_df

# Example usage:
start_time = time.time()
symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
print("Starting option trading opportunity analysis...\n")
df = load_features(symbols)
if not df.empty:
    # Filter to last 8 years only
    df = filter_last_n_years(df)
    X, y, df = prepare_data(df, target_return=0.015, target_period=5)
    # selected_features = select_features_boruta(X, y)
    # X = X[selected_features]
    selected_features = X.columns.tolist()
    model = train_naive_bayes(X, y)
    opportunities = identify_opportunities(model, df, feature_cols=selected_features)
    print("Option Trading Opportunities:")
    if not opportunities.empty:
        opportunities = opportunities.sort_values(by='Opportunity_Prob', ascending=False)
        opportunities['Opportunity_Prob'] = opportunities['Opportunity_Prob'].apply(lambda x: f"{x:.4f}")
        print(opportunities.to_string(index=False))
    else:
        print("No trading opportunities found based on the current threshold.")
        latest = df.sort_values('Date').groupby('symbol').tail(1)
        # Ensure all selected features are present in latest
        for col in X.columns:
            if col not in latest.columns:
                latest[col] = np.nan
        X_latest = latest[X.columns.tolist()]
        probs = model.predict_proba(X_latest)[:, 1]
        latest['Opportunity_Prob'] = probs
        latest = latest.sort_values(by='Opportunity_Prob', ascending=False)
        latest['Opportunity_Prob'] = latest['Opportunity_Prob'].apply(lambda x: f"{x:.4f}")
        print("\nAll symbols with their opportunity probabilities:")
        print(latest[['Date','symbol', 'Opportunity_Prob']].to_string(index=False))
    # --- Backtest using vectorbt ---
    backtest_signals(df, model, selected_features, threshold=0.7)
else:
    print("No data loaded.")

# Example usage for scenario analysis after loading df and symbols:
target_periods = [2, 3, 5, 7]
target_returns = [0.01, 0.015, 0.02]
bt_scenario = scenario_analysis(df, target_periods, target_returns, symbols, years=5, threshold=0.7)
bt_scenario.to_csv("scenario_analysis.csv",index=False)

end_time = time.time()
print(f"\nScript completed in {int(end_time - start_time)} seconds.")



