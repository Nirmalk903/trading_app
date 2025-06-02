import pandas as pd
import numpy as np
import os
from datetime import datetime
import json


dt = datetime.now()

def get_symbols(dt,top_n=17):
    dt = dt.strftime('%d-%b-%Y')
    nifty_fity_path = os.path.join('./Nifty_Fifty', "MW-NIFTY-50.csv")
    if os.path.exists(nifty_fity_path):
        nifty_fifty = pd.read_csv(nifty_fity_path)
        nifty_symbols = nifty_fifty['SYMBOL \n'].tolist()
        liquid_symbols = nifty_symbols + ['NIFTY', 'BANKNIFTY']
    else:
        print(f"File {nifty_fity_path} does not exist.")
    
    file_name = f'LA-MOST-ACTIVE-UNDERLYING-{dt}.csv'
    file_path = os.path.join('./Most_Active_Underlying', file_name)
    if os.path.exists(file_path):
        most_active = pd.read_csv(file_path)
        most_active = most_active[~most_active['Symbol'].isin(['MIDCPNIFTY', 'FINNIFTY','NIFTYIT','NIFTYNXT50','NIFTYPSUBANK','NIFTYINFRA','NIFTYMETAL','NIFTYPHARMA','NIFTYMEDIA','NIFTYAUTO','NIFTYCONSUMPTION','NIFTYENERGY','NIFTYFMCG','NIFTYHEALTHCARE'])].reset_index(drop=True)
        most_active = most_active[most_active['Symbol'].isin(liquid_symbols)].reset_index(drop=True)
        most_active['YF_Symbol'] = most_active['Symbol'].apply(lambda x: '^NSEI' if x == 'NIFTY' else '^NSEBANK' if x == 'BANKNIFTY' else f'{x}.NS')
        most_active = most_active.sort_values(by='Value (? Lakhs) - Total', ascending=False).reset_index(drop=True)
        most_active = most_active.head(top_n)
    else:
        print(f"File {file_name} does not exist.")
        
    symbols = most_active['Symbol'].tolist()
    yf_symbols = most_active['YF_Symbol'].tolist()
    return symbols, yf_symbols

symbols , yf_symbols = get_symbols(dt,top_n=17)



symbols




# Add these lines at the very top of your script for auto-reloading modules in Jupyter/IPython environments

try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
except Exception:
    pass



symbol = 'NIFTY'

symbol = "^NSEI" if symbol == 'NIFTY' else "^NSEBANK" if symbol == 'BANKNIFTY' else f'{symbol}.NS'

symbol




# ...existing code...