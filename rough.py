
import datetime as dt
from nsepython import *
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta

def get_nifty50_symbols():
    """
    Returns list of NIFTY 50 stock symbols.
    """
    import pandas as pd
    nifty50 = pd.read_html("https://www1.nseindia.com/content/indices/ind_nifty50list.csv")[0]
    return nifty50['Symbol'].tolist()

get_nifty50_symbols()

def get_atm_option_iv(symbol, expiry=None):
    """
    Fetches IV of ATM Call Option for the given symbol.
    """
    spot = nse_quote_ltp(symbol)
    print(f"{symbol}: Spot ₹{spot}")

    # Get nearest weekly/monthly expiry if not supplied
    if expiry is None:
        chain = nse_optionchain_scrapper(symbol, "equity")
        expiries = sorted(set([i['expiryDate'] for i in chain['records']['data']]))
        expiry = expiries[0]

    # Fetch entire chain for the expiry
    option_chain = nse_optionchain_scrapper(symbol, "equity")[ 'records']['data']
    strike_list = sorted({i['strikePrice'] for i in option_chain})
    atm_strike = min(strike_list, key=lambda x: abs(x - spot))

    print("  Expiry:", expiry, " | ATM strike:", atm_strike)

    # filter ATM CALL row
    atm_row = next(
        i for i in option_chain 
        if i['strikePrice'] == atm_strike and i.get('CE') and i['expiryDate'] == expiry
    )
    option_ltp = atm_row['CE']['lastPrice']

    ttm = (dt.datetime.strptime(expiry, "%d-%b-%Y") - dt.datetime.now()).days / 365.0
    r = 0.06  # assumed risk-free 6%

    iv = implied_volatility(
        price     = option_ltp,
        S         = spot,
        K         = atm_strike,
        t         = ttm,
        r         = r,
        flag      = 'c'
    )
    return round(iv * 100,2)

if __name__ == "__main__":
    symbols = get_nifty50_symbols()
    results = {}
    for sym in symbols:
        try:
            iv_atm = get_atm_option_iv(sym)
            results[sym] = iv_atm
            print(f"  → Implied Vol: {iv_atm}%")
        except Exception as e:
            print(f"{sym}: Error - {e}")

    print("\nATM IV snapshot:")
    for s, v in results.items():
        print(s, v, "%")

get_atm_option_iv("NSE:RELIANCE", "28-Aug-2025")  # Example usage