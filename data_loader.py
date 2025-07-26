import pandas as pd
import yfinance as yf

def download_stock(symbol="AAPL", start="2015-01-01", end="2020-12-31", filename="stock_prices.csv"):
    data = yf.download(symbol, start=start, end=end)
    data.to_csv(filename)
    print(f"Saved {len(data)} rows to {filename}")

if __name__ == "__main__":
    download_stock()
