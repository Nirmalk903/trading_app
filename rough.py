import pendulum as p
from datetime import datetime
from dateutil.parser import parse

date_now = p.now(tz='local')

def time_to_expiry(expiry):
    current_time = p.now(tz='local')
    # expiry = parse(expiry)
    expiry = p.datetime(expiry.year, expiry.month, expiry.day,15,30)
    delta = expiry.diff(current_time).in_days()/365
    return delta


dt = datetime.now()
dt