"""
Fetch monthly option chains (JSON/CSV) using KiteConnect.
- Picks underlyings from get_symbols() (most-active) or CLI --underlying
- Groups option instruments by expiry (MONYYYY) and strikes
- Prompts user to select expiry (or use --expiry / --all)
- Saves results under results/kite/OptionChainJSON/<UNDERLYING>/<EXPIRY>/
"""
import os
import sys
import json
import re
import time
import glob
import logging
import calendar
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

import pandas as pd

# allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_download_kite import _build_instrument_map_from_kite  # noqa: E402
from data_download_vbt import get_symbols, get_dates_from_most_active_files  # noqa: E402

load_dotenv()
logger = logging.getLogger("optionchain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")

MONTHS = {m[:3].upper(): i for i, m in enumerate(calendar.month_name) if m}  # {'JAN':1, ...}


def _extract_expiry_key_from_symbol(ts: str) -> Optional[str]:
    """Return expiry as MONYYYY (e.g. SEP2025) from a tradingsymbol, or None."""
    if not ts:
        return None
    s = str(ts).upper().replace(".", " ").replace("-", " ")
    m = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{2,4})\b", s)
    if not m:
        return None
    mon, yr = m.group(1), m.group(2)
    try:
        y = int(yr)
        if y < 100:
            y += 2000
        return f"{mon}{y}"
    except Exception:
        return None


def _group_option_tokens_by_expiry(inst_map: Dict[str, int], underlying: str) -> Dict[str, Dict[int, Dict]]:
    """
    Group option tokens by expiry:
      { expiry_key (MONYYYY) : { strike : {CE, PE, symbol_ce, symbol_pe} } }
    """
    out: Dict[str, Dict[int, Dict]] = {}
    ux = str(underlying).upper()
    pat = re.compile(r"(\d+)\s*(CE|PE)$", re.IGNORECASE)
    for ts, tok in inst_map.items():
        try:
            tss = str(ts).upper()
        except Exception:
            continue
        if ux not in tss:
            continue
        expiry = _extract_expiry_key_from_symbol(tss) or "UNKNOWN"
        m = pat.search(tss.replace(" ", ""))
        if not m:
            continue
        strike = int(m.group(1))
        side = m.group(2).upper()
        exp_map = out.setdefault(expiry, {})
        rec = exp_map.setdefault(strike, {"CE": None, "PE": None, "symbol_ce": None, "symbol_pe": None})
        if side == "CE":
            rec["CE"] = int(tok) if isinstance(tok, (int, str)) and str(tok).isdigit() else tok
            rec["symbol_ce"] = tss
        else:
            rec["PE"] = int(tok) if isinstance(tok, (int, str)) and str(tok).isdigit() else tok
            rec["symbol_pe"] = tss
    return out


def _map_ltp_by_token(kite, tokens: List[int]) -> Dict[int, Dict]:
    """Fetch kite.ltp for tokens and return mapping token -> quote dict."""
    out: Dict[int, Dict] = {}
    if not tokens:
        return out
    try:
        resp = kite.ltp(tokens)
    except Exception:
        resp = {}
        for t in tokens:
            try:
                single = kite.ltp(t)
                if isinstance(single, dict):
                    resp.update(single)
            except Exception:
                continue

    for inst_key, val in (resp or {}).items():
        if not isinstance(val, dict):
            continue
        tok = val.get("instrument_token")
        if tok is None:
            m = re.search(r"(\d+)$", str(inst_key))
            if m:
                try:
                    tok = int(m.group(1))
                except Exception:
                    tok = None
        if tok is not None:
            try:
                out[int(tok)] = val
            except Exception:
                continue
    return out


def save_chain(out_dir: str, underlying: str, expiry_key: str, chain: Dict[int, Dict], ltp_map: Dict[int, Dict]) -> None:
    """Save option chain as JSON and CSV under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for strike in sorted(chain.keys()):
        rec = chain[strike]
        ce_tok = rec.get("CE")
        pe_tok = rec.get("PE")
        ce_data = ltp_map.get(ce_tok) if ce_tok else None
        pe_data = ltp_map.get(pe_tok) if pe_tok else None

        rows.append({
            "strike": strike,
            "symbol_ce": rec.get("symbol_ce"),
            "ce_token": ce_tok,
            "ce_ltp": (ce_data.get("last_price") if ce_data else None),
            "ce_oi": (ce_data.get("oi") if ce_data else None),
            "ce_volume": (ce_data.get("volume") if ce_data else None),
            "symbol_pe": rec.get("symbol_pe"),
            "pe_token": pe_tok,
            "pe_ltp": (pe_data.get("last_price") if pe_data else None),
            "pe_oi": (pe_data.get("oi") if pe_data else None),
            "pe_volume": (pe_data.get("volume") if pe_data else None),
        })

    json_path = os.path.join(out_dir, f"{underlying}_{expiry_key}_OptionChain.json")
    csv_path = os.path.join(out_dir, f"{underlying}_{expiry_key}_OptionChain.csv")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump({"underlying": underlying, "expiry": expiry_key, "rows": rows}, f, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved option chain files: %s / %s", json_path, csv_path)


def _determine_symbol_date_from_most_active() -> datetime:
    """Return latest timestamp (tz-aware UTC) from MOST-ACTIVE*UNDERLYING files, or fallback to get_dates_from_most_active_files()/now."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pattern = os.path.join(repo_root, "**", "*MOST-ACTIVE*UNDERLYING*.csv")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        dirs: Dict[str, List[str]] = {}
        for f in matches:
            d = os.path.dirname(f)
            dirs.setdefault(d, []).append(f)
        best_dir, files = max(dirs.items(), key=lambda kv: len(kv[1]))
        latest_file = max(files, key=os.path.getmtime)
        return datetime.fromtimestamp(os.path.getmtime(latest_file), timezone.utc)
    try:
        dates = get_dates_from_most_active_files() or []
        if dates:
            return dates[-1].astimezone(timezone.utc) if hasattr(dates[-1], "tzinfo") else dates[-1].replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _prompt_select_expiry(expiry_keys: List[str], default: Optional[str] = None) -> Optional[str]:
    """Prompt user to pick one expiry from expiry_keys, or 'ALL', or skip."""
    if not expiry_keys:
        return None
    keys = sorted([k for k in expiry_keys if k != "UNKNOWN"]) + (["UNKNOWN"] if "UNKNOWN" in expiry_keys else [])
    print("\nAvailable expiries:")
    for i, k in enumerate(keys):
        default_mark = " (default)" if default and k == default else ""
        print(f"  {i:2d}) {k}{default_mark}")
    print("  a) ALL expiries")
    print("  s) SKIP this underlying")
    while True:
        sel = input(f"Select expiry index [default '{default}'] (index / a / s): ").strip()
        if sel == "" and default:
            return default
        if sel.lower() in ("a", "all"):
            return "ALL"
        if sel.lower() in ("s", "skip"):
            return None
        try:
            idx = int(sel)
            if 0 <= idx < len(keys):
                return keys[idx]
        except Exception:
            pass
        print("Invalid selection; try again.")


def _select_month_end_expiry(candidate_exps: List[str]) -> Optional[str]:
    """Automatically pick the earliest expiry whose month-end is >= now (UTC).
    candidate_exps are keys like MONYYYY (e.g. SEP2025). Returns chosen key or None.
    """
    if not candidate_exps:
        return None
    now = datetime.now(timezone.utc)
    months = {}
    for ek in candidate_exps:
        if ek == "UNKNOWN":
            continue
        try:
            mon = ek[:3]
            yr = int(ek[3:])
            mnum = MONTHS.get(mon, 0)
            if mnum <= 0:
                continue
            last_day = calendar.monthrange(yr, mnum)[1]
            expiry_dt = datetime(yr, mnum, last_day, 23, 59, 59, tzinfo=timezone.utc)
            months[ek] = expiry_dt
        except Exception:
            continue
    # pick the earliest expiry_dt >= now, otherwise the latest available
    future = sorted([(dt, ek) for ek, dt in months.items() if dt >= now], key=lambda x: x[0])
    if future:
        return future[0][1]
    if months:
        # fallback to most recent past expiry
        past = sorted([(dt, ek) for ek, dt in months.items()], key=lambda x: x[0])
        return past[-1][1]
    # if no parsed expiries, prefer UNKNOWN if present
    return "UNKNOWN" if "UNKNOWN" in candidate_exps else None


def main():
    from kiteconnect import KiteConnect  # local import
    import argparse

    parser = argparse.ArgumentParser(description="Fetch monthly option chains for underlyings")
    parser.add_argument("-u", "--underlying", help="Single underlying (overrides get_symbols list)")
    parser.add_argument("-n", "--top-n", type=int, default=50, help="Number of underlyings from get_symbols() to fetch")
    parser.add_argument("--interval-sleep", type=float, default=0.35, help="Polite sleep between kite calls (seconds)")
    parser.add_argument("--expiry", help="Expiry key to use (e.g. SEP2025) or 'ALL' to fetch all (non-interactive)")
    args = parser.parse_args()

    SESSION_PATH = os.path.expanduser("~/.kite_session.json")
    API_KEY = os.getenv("KITE_API_KEY")
    if not API_KEY or not os.path.exists(SESSION_PATH):
        logger.error("KITE_API_KEY not set or session missing (~/.kite_session.json)")
        raise SystemExit(1)
    try:
        sd = json.load(open(SESSION_PATH, "r"))
    except Exception:
        logger.error("Failed to read session file %s", SESSION_PATH)
        raise SystemExit(1)
    token = sd.get("access_token")
    if not token:
        logger.error("access_token missing in saved session")
        raise SystemExit(1)

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(token)
    try:
        kite.profile()
    except Exception as e:
        logger.exception("Kite authentication failed: %s", e)
        raise SystemExit(1)

    inst_map = _build_instrument_map_from_kite(kite, exchanges=["NFO", "NSE", "NSE_INDICES", "BSE"])
    if not inst_map:
        logger.error("Instrument map empty; aborting")
        raise SystemExit(1)
    inst_map_norm: Dict[str, int] = {}
    for k, v in inst_map.items():
        try:
            ku = str(k).upper()
            inst_map_norm[ku] = v
            inst_map_norm["".join(ch for ch in ku if ch.isalnum())] = v
        except Exception:
            continue

    if args.underlying:
        underlyings = [args.underlying.strip().upper()]
    else:
        symbol_date = _determine_symbol_date_from_most_active()
        symbols, _meta = get_symbols(symbol_date, top_n=args.top_n)
        underlyings = [str(s).upper() for s in symbols]

    if not underlyings:
        logger.info("No underlyings to process")
        return

    out_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "kite", "OptionChainJSON"))
    os.makedirs(out_base, exist_ok=True)

    for u in underlyings:
        try:
            logger.info("Processing underlying: %s", u)
            expiry_chains = _group_option_tokens_by_expiry(inst_map_norm, u)
            if not expiry_chains:
                logger.info("No option instruments found for %s; skipping.", u)
                continue

            candidate_exps = [ek for ek in expiry_chains.keys()]
            if not candidate_exps:
                logger.info("No expiries found for %s; skipping.", u)
                continue

            # automatically select month-end expiry (no user input)
            if args.expiry and args.expiry.strip().upper() != "ALL":
                chosen_expiry = args.expiry.strip().upper() if args.expiry.strip().upper() in candidate_exps else None
                if not chosen_expiry:
                    logger.warning("Requested expiry %s not found for %s; falling back to automatic selection", args.expiry, u)
            elif args.expiry and args.expiry.strip().upper() == "ALL":
                expiries_to_fetch = candidate_exps
                logger.info("Fetching ALL expiries for %s (count=%d)", u, len(expiries_to_fetch))
                # iterate below
                chosen_expiry = None
            else:
                chosen_expiry = _select_month_end_expiry(candidate_exps)
                logger.info("Auto-selected month-end expiry %s for %s", chosen_expiry, u)

            expiries_to_fetch = expiries_to_fetch if 'expiries_to_fetch' in locals() and expiries_to_fetch else ([chosen_expiry] if chosen_expiry else [])
            if not expiries_to_fetch:
                logger.info("No expiries selected for %s; skipping.", u)
                continue

            for chosen_expiry in expiries_to_fetch:
                logger.info("Fetching expiry %s for %s", chosen_expiry, u)
                chain = expiry_chains.get(chosen_expiry) or {}
                if not chain:
                    logger.info("No strikes for expiry %s for %s; skipping.", chosen_expiry, u)
                    continue

                tokens = sorted({int(t) for rec in chain.values() for t in (rec.get("CE"), rec.get("PE")) if t})
                ltp_map: Dict[int, Dict] = {}
                batch_size = 100
                for i in range(0, len(tokens), batch_size):
                    batch = tokens[i: i + batch_size]
                    try:
                        ltp_map.update(_map_ltp_by_token(kite, batch))
                    except Exception as e:
                        logger.debug("ltp batch failed for %s %s: %s", u, chosen_expiry, e)
                    time.sleep(args.interval_sleep)

                out_dir = os.path.join(out_base, u, chosen_expiry)
                save_chain(out_dir, u, chosen_expiry, chain, ltp_map)
                time.sleep(args.interval_sleep)
        except Exception as e:
            logger.exception("Failed for %s: %s", u, e)
            time.sleep(args.interval_sleep)


if __name__ == "__main__":
    main()