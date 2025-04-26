

# from helium import *

# # Start Chrome and navigate to the NSE option chain page
# start_chrome("https://www.nseindia.com/option-chain")

# # Wait for the page to load
# wait_until(Text("Option Chain").exists)

# # Click on the dropdown to select the index
# click("Select Option Type")

# # Write "NIFTY" and press Enter to select it
# write("NIFTY", into="Select Option Type")
# press(ENTER)

# # Verify if NIFTY is selected
# wait_until(Text("NIFTY").exists)

# print("NIFTY selected successfully!")