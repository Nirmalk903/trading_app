from Options_Utility import *
from get_data import enrich_option_chain, fetch_live_options_data, fetch_and_save_options_chain
from quantlib_black_scholes import calculate_implied_volatility
from tenacity import retry, wait_random, stop_after_attempt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

url = "https://www.nseindia.com/market-data/most-active-underlying"

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'connection':'keep-alive',
    'sec-fetch-mode':'cors',
    'Referrer Policy':'strict-origin-when-cross-origin',
    'referer': 'https://www.nseindia.com/market-data/most-active-underlying',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0'}
response = requests.get(url,headers=headers)
# response = requests.get(url)
print(response.status_code)
if response.status_code == 200:
    print("Page fetched successfully")
else:
    print("Failed to fetch the page")


soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'class': 'most-active-table'})
print(table)
# # rows = table.find_all('tr')[1:]  # Skip the header row  




# @retry(wait=wait_random(min=0.1, max=1))
@retry(wait=wait_random(min=1, max=3), stop=stop_after_attempt(50))
def fetch_most_active_und():
    """
    Fetch the most active underlying data from NSE and save it to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the most active underlying data.
    """
    url = "https://www.nseindia.com/api/live-analysis-most-active-underlying"
    
    # cookie = "bm_sv=685D1C1813FEF99DC08E90EA691BB315~YAAQxVI2F73wmT+WAQAAx0yBYRsfZ7iSwWfElscx/rRS3LqItaqzd0yO7HTZoowTjOQjwm8Qjjrqz80duMAyWG+5uzI+JRaUWhTPaV5SWuGRX6pXcQTuXczZ9xi6I+jp5aK52eOBx5APu71hY5elvqBYZPJzo+0Y8dD3aklrZNRcSnrmW+OfBkf08/EPJnhJwN1GKKcAQGZCywb61Lx3wIL2RPly/zOHlkLemvTIEezvG8x6I40Wiu2yVtrP3fDjo1u7~1; Domain=.nseindia.com; Path=/; Expires=Wed, 23 Apr 2025 09:13:26 GMT; Max-Age=7021; Secure"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/market-data/most-active-underlying",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    }

    try:
        # Initial request to fetch cookies
        session = requests.Session()
        response = session.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Initial request failed with status code {response.status_code}")
            return None

        # Use cookies from the initial response
        cookies = dict(response.cookies)

        # Second request with cookies
        response = session.get(url, headers=headers, cookies=cookies)
        if response.status_code != 200:
            print(f"Second request failed with status code {response.status_code}")
            return None

        # Parse JSON response
        data = response.json()
        if 'data' not in data:
            print("No 'data' key found in the response")
            return None

        # Convert to DataFrame and process
        df = pd.DataFrame(data.get('data'))
        df = df.sort_values(by=['totVolume'], ascending=False).reset_index(drop=True).head(30)

        # Save to CSV
        df.to_csv('most_active_underlying.csv', index=False)
        print("Data saved to 'most_active_underlying.csv'")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return None
    
fetch_most_active_und()