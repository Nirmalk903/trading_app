# ...existing code...
import os
import urllib.parse
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Optional, List
from bs4 import BeautifulSoup
import requests
import requests.utils
from selenium import webdriver
import browser_cookie3

def download_most_active_csv(out_dir: Optional[str] = None, timeout: float = 15.0) -> Optional[str]:
    """Robust download for MOST-ACTIVE-UNDERLYING CSV from NSE.
    - Performs cookie preflight against https://www.nseindia.com
    - Calls API endpoint /api/most-active-underlying and converts JSON -> CSV
    - Retries with backoff. If NSE blocks (403) suggest using a browser or Selenium.
    """
    out_dir = out_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Most_Active_Underlying"))
    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    # Do not hardcode a long cookie string inline (it may contain unescaped quotes and break the parser).
    # If you need to supply cookies, either:
    #  - Use browser_cookie3: inject_cookies_from_browser(session)
    #  - Use Selenium and inject_cookies_from_selenium(session, driver)
    #  - Or paste a simple Cookie header string here (make sure to escape quotes), e.g.:
    #      inject_cookie_header(session, 'ak_bmsc=...; other_cookie=...') 
    base_url = "https://www.nseindia.com"
    page_url = "https://www.nseindia.com/market-data/most-active-underlying"

    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": page_url,
        "Origin": "https://www.nseindia.com",
        "Connection": "keep-alive",
    }
    session.headers.update(base_headers)

    # helper to save bytes / dataframe to file
    def _save_bytes(content: bytes, src_name: str) -> str:
        name = os.path.basename(urllib.parse.urlparse(src_name).path) or "most_active_underlying.csv"
        fname = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{name}"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as fh:
            fh.write(content)
        return out_path

    def _save_df(df: pd.DataFrame) -> str:
        fname = f"most_active_underlying_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = os.path.join(out_dir, fname)
        df.to_csv(out_path, index=False)
        return out_path

    # 1) preflight: visit base and page to obtain cookies; retry with backoff
    r = None
    for attempt in range(1, 7):
        try:
            session.get(base_url, timeout=timeout)
            time.sleep(0.5)
            r = session.get(page_url, timeout=timeout)
            if r is not None and r.status_code == 200 and r.text:
                break
            # if blocked, wait and retry
        except Exception:
            pass
        sleep_t = min(2 ** attempt * 0.1, 4.0)
        time.sleep(sleep_t)

    if not r or r.status_code != 200:
        print("Landing page fetch failed (status=%s). NSE may block programmatic requests." % (getattr(r, "status_code", None),))
        print("Try opening https://www.nseindia.com in a browser once (to set cookies) or use Selenium.")
        return None

    # 2) try API endpoint returning JSON (preferred)
    api_json = "https://www.nseindia.com/api/most-active-underlying"
    headers_json = dict(session.headers)
    headers_json.update({"Accept": "application/json, text/javascript, */*; q=0.01", "X-Requested-With": "XMLHttpRequest"})
    try:
        resp = session.get(api_json, headers=headers_json, timeout=timeout)
        if resp.status_code == 200:
            ctype = resp.headers.get("Content-Type", "").lower()
            # if CSV-like response
            if "csv" in ctype or api_json.endswith(".csv"):
                if resp.content:
                    return _save_bytes(resp.content, api_json)
            # otherwise try parse JSON
            text = resp.text.strip()
            if text.startswith("{") or text.startswith("[") or "application/json" in ctype:
                j = None
                try:
                    j = resp.json()
                except Exception:
                    import re
                    m = re.search(r"(\{.*\}|\[.*\])", resp.text, flags=re.S)
                    if m:
                        j = json.loads(m.group(1))
                if j:
                    rows = None
                    if isinstance(j, dict):
                        for k in ("data", "records", "result", "rows", "underlying"):
                            if k in j and isinstance(j[k], list):
                                rows = j[k]
                                break
                    elif isinstance(j, list):
                        rows = j
                    if rows:
                        df = pd.DataFrame(rows)
                        return _save_df(df)
        elif resp.status_code in (401, 403):
            # blocked by NSE
            print("API returned status %d (blocked). NSE often blocks scripts; try opening the site in a browser once or use Selenium." % resp.status_code)
        # continue to try other options
    except Exception:
        pass

    # 3) try legacy CSV endpoint or discovered CSV links on landing HTML
    # check known CSV endpoint
    csv_candidate = "https://www.nseindia.com/live_market/dynaContent/live_watch/most_active_underlying.csv"
    try:
        rr = session.get(csv_candidate, timeout=timeout)
        if rr.status_code == 200 and rr.content:
            return _save_bytes(rr.content, csv_candidate)
    except Exception:
        pass

    # 4) try to find any .csv link on landing page (HTML anchors)
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href and ".csv" in href.lower():
                csv_url = urllib.parse.urljoin(page_url, href)
                try:
                    rr = session.get(csv_url, timeout=timeout)
                    if rr.status_code == 200 and rr.content:
                        return _save_bytes(rr.content, csv_url)
                except Exception:
                    continue
    except Exception:
        pass

    # 5) fallback: try to parse any HTML table on the landing page (may be empty due to JS)
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr"):
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cols:
                    rows.append(cols)
            if rows:
                if headers and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(rows)
                return _save_df(df)
    except Exception:
        pass

    print("Unable to locate CSV or table in page. NSE often blocks programmatic requests (403).")
    print("Workarounds:")
    print("  1) open the page once in your browser to set cookies, then try again;")
    print("  2) use Selenium to render the page, then save CSV;")
    print("  3) run this script from a machine/IP that NSE allows.")
    return None

def inject_cookies_from_browser(session: requests.Session) -> None:
    """Try to load cookies from the local browser (requires browser_cookie3)."""
    try:
        import browser_cookie3
    except Exception:
        return
    try:
        cj = browser_cookie3.chrome(domain_name="nseindia.com")  # or .firefox()
        session.cookies.update(requests.utils.dict_from_cookiejar(cj))
    except Exception:
        return

def inject_cookies_from_selenium(session: requests.Session, driver) -> None:
    """Transfer cookies from a Selenium webdriver instance into requests.Session."""
    for c in driver.get_cookies():
        # set cookie in session; domain/path may be used by server
        session.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path"))

def inject_cookie_header(session: requests.Session, cookie_header: str) -> None:
    """Set a raw Cookie header (manual copy from browser devtools)."""
    session.headers.update({"Cookie": cookie_header})

if __name__ == "__main__":
    path = download_most_active_csv()
    if path:
        print("Saved:", path)
    else:
        print("Saved: None")
