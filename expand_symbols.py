import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print("Fetching S&P 500...")
sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
r = requests.get(sp500_url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')
table = soup.find('table', {'id': 'constituents'})

sp500_tickers = []
for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if cols:
        ticker = cols[0].text.strip()
        sp500_tickers.append(ticker)

print("Fetching S&P 400 MidCap...")
sp400_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
r2 = requests.get(sp400_url, headers=headers)
soup2 = BeautifulSoup(r2.text, 'html.parser')
table2 = soup2.find('table', {'id': 'constituents'})

sp400_tickers = []
for row in table2.find_all('tr')[1:]:
    cols = row.find_all('td')
    if cols:
        ticker = cols[0].text.strip()
        sp400_tickers.append(ticker)

core_nodes = [
    "SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "SLV", "USO", "UNG", "UUP",
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X", "ILS=X",
    "BTC-USD", "ETH-USD",
    "CL=F", "GC=F", "SI=F", "HG=F",
    "TSM", "ASML", "NVO", "NVS", "BABA", "TM", "TTE"
]

all_tickers = list(set(sp500_tickers + sp400_tickers + core_nodes))
all_tickers = [str(t).replace('.', '-') for t in all_tickers]
all_tickers = sorted(all_tickers)

print(f"Total unique tickers fetched: {len(all_tickers)}")

with open("market_analyzer.py", "r") as f:
    content = f.read()

symbols_list_str = "SYMBOLS = [\n    " + ",\n    ".join([f'"{t}"' for t in all_tickers]) + "\n]"
content = re.sub(r'SYMBOLS = \[.*?\]', symbols_list_str, content, flags=re.DOTALL)

with open("market_analyzer.py", "w") as f:
    f.write(content)
