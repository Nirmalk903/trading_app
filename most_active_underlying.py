from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

download_dir = './Most_Active_Underlying'
os.makedirs(download_dir, exist_ok=True)

service = Service(r'C:\chromedriver\chromedriver.exe')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# url = "https://www.nseindia.com/market-data/most-active-underlying"

url = "https://www.nseindia.com/market-data/oi-spurts"

driver.get(url)
time.sleep(10)  # Wait for JS to load the table

# Accept cookies if prompted
try:
    accept_button = driver.find_element(By.ID, "accept-cookie-policy")
    accept_button.click()
except:
    pass

# Find the table and extract HTML
table = driver.find_element(By.TAG_NAME, "table")
html = table.get_attribute('outerHTML')
driver.quit()

# Parse with pandas
df = pd.read_html(html)[0]
df.to_csv(os.path.join(download_dir, "most_active.csv"), index=False)
print("Downloaded complete table to Most_Active_Underlying/most_active.csv")