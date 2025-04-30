import QuantLib as ql
import pandas as pd



# Set up QuantLib objects
calendar = ql.NullCalendar()
day_count = ql.Actual365Fixed()
settlement_date = ql.Date.todaysDate()
time_to_expiry = 2/365
maturity_date = settlement_date + int(time_to_expiry * 365)
maturity_date