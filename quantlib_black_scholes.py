import QuantLib as ql

def calculate_implied_volatility(option_price, spot_price, strike_price, risk_free_rate, time_to_expiry, option_type):
    """
    Calculate the implied volatility of an option using QuantLib.

    Args:
        option_price (float): The market price of the option.
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        risk_free_rate (float): The risk-free interest rate (annualized).
        time_to_expiry (float): Time to expiry in years.
        option_type (str): 'call' or 'put'.

    Returns:
        float: The implied volatility.
    """
    # Define option type
    if option_type.lower() == 'call':
        option_type = ql.Option.Call
    elif option_type.lower() == 'put':
        option_type = ql.Option.Put
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Set up QuantLib objects
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    settlement_date = ql.Date.todaysDate()
    maturity_date = settlement_date + int(time_to_expiry * 365)

    # Option payoff and exercise
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)

    # Create the option
    european_option = ql.VanillaOption(payoff, exercise)

    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, risk_free_rate, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, 0.2, day_count))  # Initial guess for volatility

    # Black-Scholes-Merton process
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, rate_handle, rate_handle, vol_handle)

    # Implied volatility calculation
    implied_vol = european_option.impliedVolatility(option_price, bsm_process)

    return implied_vol * 100 # Return as a float
    # Check if the implied volatility is valid 


# # Example usage
# option_price = 10.0  # Market price of the option
# spot_price = 100.0   # Current price of the underlying asset
# strike_price = 105.0 # Strike price
# risk_free_rate = 0.05 # Annualized risk-free rate
# time_to_expiry = 0.5  # Time to expiry in years
# option_type = 'call'  # Option type ('call' or 'put')

# implied_vol = calculate_implied_volatility(option_price, spot_price, strike_price, risk_free_rate, time_to_expiry, option_type)
# print(f"Implied Volatility: {implied_vol:.2%}")



import pandas as pd

# Assuming `calculate_implied_volatility` is already defined

def apply_implied_volatility(option_chain):
    """
    Apply the implied volatility calculation to an option chain.

    Args:
        option_chain (pd.DataFrame): A DataFrame containing the option chain data with columns:
                                     ['option_price', 'spot_price', 'strike_price', 'risk_free_rate', 'time_to_expiry', 'option_type'].

    Returns:
        pd.DataFrame: The updated DataFrame with an additional 'implied_volatility' column.
    """
    # Initialize an empty list to store implied volatilities
    implied_vols = []

    # Iterate over each row in the DataFrame
    for _, row in option_chain.iterrows():
        try:
            # Calculate implied volatility for each row
            iv = calculate_implied_volatility(
                option_price=row['option_price'],
                spot_price=row['spot_price'],
                strike_price=row['strike_price'],
                risk_free_rate=row['risk_free_rate'],
                time_to_expiry=row['time_to_expiry'],
                option_type=row['option_type']
            )
        except Exception as e:
            # Handle cases where implied volatility cannot be calculated
            print(f"Error calculating IV for row {row}: {e}")
            iv = None

        # Append the result to the list
        implied_vols.append(iv)

    # Add the implied volatilities as a new column in the DataFrame
    option_chain['implied_volatility'] = implied_vols

    return option_chain


