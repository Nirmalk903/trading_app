import pandas as pd
import numpy as np
import os
from datetime import datetime
import json


dt = datetime.now()

def get_symbols(dt,top_n=17):
    dt = dt.strftime('%d-%b-%Y')
    file_name = f'LA-MOST-ACTIVE-UNDERLYING-{dt}.csv'
    file_path = os.path.join('./Most_Active_Underlying', file_name)
    if os.path.exists(file_path):
        most_active = pd.read_csv(file_path)
        most_active = most_active[~most_active['Symbol'].isin(['MIDCPNIFTY', 'FINNIFTY','NIFTYIT','NIFTYNXT50','NIFTYPSUBANK','NIFTYINFRA','NIFTYMETAL','NIFTYPHARMA','NIFTYMEDIA','NIFTYAUTO','NIFTYCONSUMPTION','NIFTYENERGY','NIFTYFMCG','NIFTYHEALTHCARE'])].reset_index(drop=True)
        # most_active['YF_Symbol'] = most_active['Symbol'].apply(lambda x: f'{x}.NS' if x not in ['^NSEI', '^NSEBANK'] else x)
        most_active['YF_Symbol'] = most_active['Symbol'].apply(lambda x: '^NSEI' if x == 'NIFTY' else '^NSEBANK' if x == 'BANKNIFTY' else f'{x}.NS')
        most_active = most_active.sort_values(by='Value (? Lakhs) - Total', ascending=False).reset_index(drop=True)
        most_active = most_active.head(top_n)
    else:
        print(f"File {file_name} does not exist.")
        
    symbols = most_active['Symbol'].tolist()
    yf_symbols = most_active['YF_Symbol'].tolist()
    return symbols, yf_symbols

symbols , yf_symbols = get_symbols(dt,top_n=17)

