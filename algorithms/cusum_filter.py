# The CUSUM filter is a sequential analysis technique used for monitoring change detection. It is particularly useful in time series analysis for identifying shifts in the mean level of a process. it's interpreted as a signal of market momentum or regime shift.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files

def getTEvents(gRaw, h):

    # ensure gRaw is a Series
    g = pd.Series(gRaw).astype(float).copy()
    diff = g.diff().fillna(0.0)

    tEvents = []
    sPos, sNeg = 0.0, 0.0
    rows = []

    for idx in diff.index[1:]:
        d = float(diff.loc[idx])
        sPos = max(0.0, sPos + d)
        sNeg = min(0.0, sNeg + d)

        trigger_val = 0
        if sNeg < -h:
            trigger_val = -1
            tEvents.append(idx)
            sNeg = 0.0
        elif sPos > h:
            trigger_val = 1
            tEvents.append(idx)
            sPos = 0.0

        rows.append({"sPos": sPos, "sNeg": sNeg, "trigger": trigger_val})

    diag_df = pd.DataFrame(rows, index=diff.index[1:])
    return pd.DatetimeIndex(tEvents), diag_df

def plot_cusum_events_all(symbols, engineered_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    cusum_dir = os.path.join(results_dir, "cusum")
    os.makedirs(cusum_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "cusum_events_close_charts.pdf")

    with PdfPages(pdf_path) as pdf:
        for symbol in symbols:
            file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
            if not os.path.exists(file_path):
                print(f"[CUSUM] missing engineered file for {symbol}, skipping")
                continue

            try:
                df = pd.read_json(file_path, orient='records', lines=True)
            except Exception as e:
                print(f"[CUSUM] failed reading {file_path}: {e}")
                continue

            if "Date" not in df.columns or "Close" not in df.columns:
                print(f"[CUSUM] {symbol} missing Date/Close, skipping")
                continue

            df = df.sort_values("Date")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
            df = df[~df.index.duplicated(keep='first')]

            # restrict to last two years only
            end_date = df.index.max()
            if pd.isna(end_date):
                continue
            start_date = end_date - pd.DateOffset(years=2)
            work_df = df.loc[start_date:end_date].copy()
            if work_df.empty:
                print(f"[CUSUM] {symbol} no data in last 2 years, skipping")
                continue

            # compute scalar threshold h
            try:
                if "garch_vol" in work_df.columns and work_df["garch_vol"].notna().any():
                    avg_garch = float(work_df["garch_vol"].mean())
                    daily_vol_frac = avg_garch / np.sqrt(252) if avg_garch > 0 else 0.0
                    h = daily_vol_frac * work_df["Close"].mean() * 0.02
                    if not np.isfinite(h) or h <= 0:
                        h = work_df["Close"].std() * 0.2
                else:
                    h = work_df["Close"].std() * 0.2
                if not np.isfinite(h) or h <= 0:
                    h = 1e-6
            except Exception:
                h = max(1e-6, work_df["Close"].std() * 0.2 if "Close" in work_df.columns else 1.0)

            t_events, diag = getTEvents(work_df["Close"], h)

            # prepare events dataframe (include diagnostics). keep consistent columns even if empty.
            if len(t_events):
                ev_close = [work_df.loc[t, "Close"] if t in work_df.index else np.nan for t in t_events]
                ev_sPos = [diag.loc[t, "sPos"] if t in diag.index else np.nan for t in t_events]
                ev_sNeg = [diag.loc[t, "sNeg"] if t in diag.index else np.nan for t in t_events]
                ev_trig = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                events_df = pd.DataFrame({
                    "event_time": list(t_events),
                    "close": ev_close,
                    "sPos": ev_sPos,
                    "sNeg": ev_sNeg,
                    "trigger": ev_trig
                })
            else:
                events_df = pd.DataFrame(columns=["event_time", "close", "sPos", "sNeg", "trigger"])

            csv_path = os.path.join(cusum_dir, f"{symbol}_cusum_events.csv")
            try:
                events_df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"[CUSUM] failed to write CSV for {symbol}: {e}")

            # plotting
            plt.figure(figsize=(12, 6))
            plt.plot(work_df.index, work_df["Close"], label="Close Price", lw=0.8)
            if len(t_events):
                triggers = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                pos_times = [t for t, tr in zip(t_events, triggers) if tr == 1]
                neg_times = [t for t, tr in zip(t_events, triggers) if tr == -1]
                if pos_times:
                    plt.scatter(pos_times, work_df.loc[pos_times, "Close"], color='green', label="CUSUM +1", zorder=5)
                if neg_times:
                    plt.scatter(neg_times, work_df.loc[neg_times, "Close"], color='red', label="CUSUM -1", zorder=5)
            plt.title(f"{symbol} - Close Price with CUSUM Events")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"All Close price charts have been saved to {pdf_path}")
    print(f"Saved per-symbol CUSUM CSV files to {cusum_dir}")


def plot_garch_cusum_events_all(symbols, engineered_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    cusum_dir = os.path.join(results_dir, "cusum")
    os.makedirs(cusum_dir, exist_ok=True)
    pdf_path = os.path.join(results_dir, "cusum_events_garch_vol_charts.pdf")

    with PdfPages(pdf_path) as pdf:
        for symbol in symbols:
            file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
            if not os.path.exists(file_path):
                print(f"[CUSUM-GARCH] missing engineered file for {symbol}, skipping")
                continue

            try:
                df = pd.read_json(file_path, orient='records', lines=True)
            except Exception as e:
                print(f"[CUSUM-GARCH] failed reading {file_path}: {e}")
                continue

            if "Date" not in df.columns or "garch_vol" not in df.columns:
                print(f"[CUSUM-GARCH] {symbol} missing Date/garch_vol, skipping")
                continue

            df = df.sort_values("Date")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
            df = df[~df.index.duplicated(keep='first')]

            # restrict to last two years only
            end_date = df.index.max()
            if pd.isna(end_date):
                continue
            start_date = end_date - pd.DateOffset(years=2)
            work_df = df.loc[start_date:end_date].copy()
            if work_df.empty:
                print(f"[CUSUM-GARCH] {symbol} no data in last 2 years, skipping")
                continue

            h = work_df["garch_vol"].std() * 0.25
            if not np.isfinite(h) or h <= 0:
                h = 1e-6

            t_events, diag = getTEvents(work_df["garch_vol"], h)

            if len(t_events):
                ev_gv = [work_df.loc[t, "garch_vol"] if t in work_df.index else np.nan for t in t_events]
                ev_sPos = [diag.loc[t, "sPos"] if t in diag.index else np.nan for t in t_events]
                ev_sNeg = [diag.loc[t, "sNeg"] if t in diag.index else np.nan for t in t_events]
                ev_trig = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                events_df = pd.DataFrame({
                    "event_time": list(t_events),
                    "garch_vol": ev_gv,
                    "sPos": ev_sPos,
                    "sNeg": ev_sNeg,
                    "trigger": ev_trig
                })
            else:
                events_df = pd.DataFrame(columns=["event_time", "garch_vol", "sPos", "sNeg", "trigger"])

            csv_path = os.path.join(cusum_dir, f"{symbol}_garch_cusum_events.csv")
            try:
                events_df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"[CUSUM-GARCH] failed to write CSV for {symbol}: {e}")

            plt.figure(figsize=(12, 6))
            plt.plot(work_df.index, work_df["garch_vol"], label="GARCH Volatility", lw=0.8)
            if len(t_events):
                triggers = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                pos_times = [t for t, tr in zip(t_events, triggers) if tr == 1]
                neg_times = [t for t, tr in zip(t_events, triggers) if tr == -1]
                if pos_times:
                    plt.scatter(pos_times, work_df.loc[pos_times, "garch_vol"], color='green', label="CUSUM +1", zorder=5)
                if neg_times:
                    plt.scatter(neg_times, work_df.loc[neg_times, "garch_vol"], color='red', label="CUSUM -1", zorder=5)
            plt.title(f"{symbol} - GARCH Volatility with CUSUM Events")
            plt.xlabel("Date")
            plt.ylabel("GARCH Volatility")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"All GARCH Volatility charts have been saved to {pdf_path}")
    print(f"Saved per-symbol GARCH CUSUM CSV files to {cusum_dir}")

if __name__ == "__main__":
    # simple console logging
    def info(msg, *args):
        print("[CUSUM]", msg % args if args else msg)

    engineered_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Engineered_data"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    # discover symbols robustly
    symbols = get_symbols(get_dates_from_most_active_files()[-1],top_n=20)[0]
    info("Processing %d symbols; engineered_dir=%s results_dir=%s", len(symbols), engineered_dir, results_dir)

    # run plotting & CSV export for CUSUM on Close and GARCH vol
    try:
        plot_cusum_events_all(symbols, engineered_dir, results_dir)
    except Exception as e:
        info("plot_cusum_events_all failed: %s", e)

    # try:
    #     plot_garch_cusum_events_all(symbols, engineered_dir, results_dir)
    # except Exception as e:
    #     info("plot_garch_cusum_events_all failed: %s", e)

    info("Done. Check %s for CSVs and charts", results_dir)