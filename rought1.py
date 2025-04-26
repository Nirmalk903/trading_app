import requests
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, wait_random, stop_after_attempt

@retry(wait=wait_random(min=1, max=3), stop=stop_after_attempt(100))
def fetch_most_active_underlying():
    """
    Fetch the most active underlying data from NSE and save it to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the most active underlying data.
    """
    base_url = "https://www.nseindia.com"
    # url = "https://www.nseindia.com/api/live-analysis-most-active-underlying"
    url = "https://www.nseindia.com/market-data/most-active-underlying"

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
        # Create a session to manage cookies
        session = requests.Session()

        # Fetch cookies from the base URL
        session.get(base_url, headers=headers)

        # Fetch the most active underlying page
        response = session.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        print(soup.prettify())

        # Find the table containing the most active underlying data
        table = soup.find({"class": "most-active-table"})
        print(table)
        if table is None:
            print("Failed to find the table in the page.")
            return None

        # Extract table rows
        rows = table.find_all("tr")[1:]  # Skip the header row
        data = []
        for row in rows:
            cols = row.find_all("td")
            cols = [col.text.strip() for col in cols]
            data.append(cols)

        # Convert to DataFrame
        columns = ["Symbol", "Volume", "Value", "Open Interest", "Change in OI"]
        df = pd.DataFrame(data, columns=columns)

        # Save to CSV
        df.to_csv("most_active_underlying.csv", index=False)
        print("Data saved to 'most_active_underlying.csv'")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Fetch and display the data
df = fetch_most_active_underlying()
if df is not None:
    print(df.head())