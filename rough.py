import pendulum as p
from datetime import datetime
from dateutil.parser import parse
from Options_Utility import *
from quantlib_black_scholes import calculate_implied_volatility
from get_data import enrich_option_chain

chain = enrich_option_chain('NIFTY')
chain['tau'] = chain['Expiry'].apply(lambda x: tau(x))
chain['rate'] = 0.1
chain

chain['atm_strike'] = chain['spot_price'].apply(lambda x: atm_strike(x,chain))

chain


def calculate_iv(row):
    if row['call_ltp'] == 0:
        return 0
    try:
        return calculate_implied_volatility(
            option_price=row['call_ltp'],
            spot_price=row['spot_price'],
            strike_price=row['atm_strike'],
            risk_free_rate=row['rate'],
            time_to_expiry=row['tau'],
            option_type='call'
        )
    except Exception as e:
        print(f"Error calculating IV for row {row}: {e}")
        return None

chain['implied_volatility'] = chain.apply(calculate_iv, axis=1)
chain