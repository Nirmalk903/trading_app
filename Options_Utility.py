import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import datetime, timedelta, time


# determining ATM strike price based on the underlying price and the option type

def atm_strike(spot,option_chain):
    multiplier = int(option_chain['strike_price'].diff().unique()[-1])
  
    # Smaller multiple
    a = (spot // multiplier) * multiplier
    # Larger multiple
    b = a + multiplier
    # Return of closest of two
    return int((b if spot - a > b - spot else a))

# Function to determine time to expiration in years

def time_to_expiry(expiry):
    current_time = datetime.datetime.now()
    expiry = parse(expiry)
    expiry = datetime.datetime(expiry.year, expiry.month, expiry.day,15,30)
    timedelta =  expiry -current_time
    tau = timedelta.total_seconds()/60
    minutes_in_year = 365*24*60
    tau = tau/minutes_in_year

    return np.round(tau, 4)

# Function to highlight dataframe rows based on a condition
def highlight_rows(x):
    if x['STRIKE PRICE'] == atm_strike:
        return['background-color: pink']*x.shape[0]
    else:
        return['background-color: white']*x.shape[0]


# Placeholder for future implementation
def main():
    print("Options Utility Script")

if __name__ == "__main__":
    main()