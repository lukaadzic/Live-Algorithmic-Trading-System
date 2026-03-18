import os

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def fetch_data(tickers, start_date, end_date):
    print(f"Fetching data for {tickers}...")
    # `auto_adjust=False` avoids some yfinance warnings/issues across versions if Close vs Adj Close
    df = yf.download(tickers, start=start_date, end=end_date)
    # yfinance often returns a MultiIndex if multiple tickers. Let's handle safely.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            df = df["Close"]
        elif "Price" in df.columns.names:
            df = df.xs("Close", level="Price", axis=1)

    # If a single ticker was passed, 'Close' is just a column, but here we have multiple
    if "Close" in df.columns and not isinstance(df.columns, pd.MultiIndex):
        # Fallback if download returned flat for some reason
        df = df[["Close"]]
    return df


def calculate_rolling_average(data, window=30):
    return data.rolling(window=window).mean()


def main():
    # Setup directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    tickers = ["AAPL", "MSFT", "AMZN", "NVDA"]
    # Download 2 years of data
    df = fetch_data(tickers, "2022-01-01", "2024-01-01")

    # Calculate rolling averages
    ma_df = pd.DataFrame()
    for col in df.columns:
        ma_df[col] = calculate_rolling_average(df[col], window=50)

    # Save to CSV
    df.to_csv("data/historical_prices.csv")
    ma_df.to_csv("data/rolling_averages.csv")

    # Plotting AAPL as an example
    plt.figure(figsize=(10, 6))
    if "AAPL" in df.columns:
        plt.plot(df.index, df["AAPL"], label="AAPL Price", color="blue")
        plt.plot(df.index, ma_df["AAPL"], label="AAPL 50-Day MA", color="orange")
    plt.title("AAPL Price vs 50-Day Rolling Average")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/price_vs_ma.png")
    print("Saved plot to plots/price_vs_ma.png")


if __name__ == "__main__":
    main()
