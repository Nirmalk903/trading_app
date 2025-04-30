import investpy #used for importing global economic calendar
import pandas as pd
import numpy as np
import pendulum as pm
import json
import yfinance as yf

def economic_calendar():
    from_date=pm.now().subtract(days=1).strftime('%d/%m/%Y')
    to_date=pm.now().add(days=3).strftime('%d/%m/%Y')
    calendar = investpy.news.economic_calendar(time_zone=None, from_date=from_date, to_date=to_date)
    calendar = calendar.drop(columns=['id','time'])
    xccy = ['USD','GBP','EUR','INR','JPY','CNH']
    importance_level = ['high']
    df = calendar.query("importance in @importance_level & currency in @xccy or zone=='india'").reset_index(drop=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.title()
        df['currency'] = df['currency'].str.upper()
    df.columns = [col.title() for col in df.columns]
    df.style.format({'Date':'{:%b %Y}'})
    df.set_index('Date')
        
        
    return df





def stock_earnings_calendar(tickers):
    from_date = pm.now().subtract(days=1)  # .strftime('%d/%m/%Y')
    to_date = pm.now().add(days=90)  # .strftime('%d/%m/%Y')

    ls = []
    for ticker in tickers:
        # print(f"Fetching earnings calendar for {ticker} from {from_date} to {to_date}")
        try:
            # Fetch the stock data
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates

            if earnings.empty:
                print(f"No earnings data available for {ticker}.")
                continue  # Skip to the next ticker

            # Format and clean up the DataFrame
            earnings.reset_index(inplace=True)
            earnings.columns = ['Date', 'EPS Estimate', 'Reported EPS', 'Surprise (%)']
            earnings['Date'] = pd.to_datetime(earnings['Date'], errors='coerce')
            earnings = earnings.query("Date >= @from_date & Date <= @to_date").reset_index(drop=True)
            earnings = earnings.sort_values(by='Date').reset_index(drop=True)
            earnings['Ticker'] = ticker  # Add the ticker symbol to the DataFrame
            ls.append(earnings)
        except Exception as e:
            print(f"Error fetching earnings calendar for {ticker}: {e}")
            continue  # Skip to the next ticker

    if not ls:
        print("No earnings data found for the provided tickers.")
        return pd.DataFrame()

    df = pd.concat(ls, ignore_index=True)
    df.set_index('Ticker', inplace=True)
    df.sort_values(by='Date', inplace=True)
    
    return df
