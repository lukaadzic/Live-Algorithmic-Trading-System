import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import pipeline


def mock_news_data(dates, tickers):
    # Mock headlines
    headlines = [
        "Record profits announced, exceeding expectations.",
        "Supply chain issues cause production delays.",
        "CEO steps down amidst controversy.",
        "New product launch receives overwhelmingly positive reviews.",
        "Regulatory hurdles might impact future earnings.",
        "Analysts upgrade stock to strong buy.",
        "Global market selloff drags down shares.",
        "Company acquires promising startup.",
        "Earnings miss expectations, stock plunges.",
    ]

    np.random.seed(42)
    news_list = []

    for date in dates:
        for ticker in tickers:
            if (
                np.random.random() > 0.8
            ):  # 20% chance of news on a given day for a ticker
                headline = np.random.choice(headlines)
                news_list.append({"Date": date, "Ticker": ticker, "Headline": headline})

    return pd.DataFrame(news_list)


def main():
    os.makedirs("plots", exist_ok=True)

    if not os.path.exists("data/historical_prices.csv"):
        print("Data not found. Run rolling_average.py first.")
        return

    prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)

    # We will test on just one ticker to make the plot clean, preferably AAPL
    ticker = "AAPL" if "AAPL" in prices.columns else prices.columns[0]
    stock_prices = prices[[ticker]].dropna()
    stock_returns = stock_prices.pct_change().dropna()

    print(f"Generating mock news for {ticker}...")
    news_df = mock_news_data(stock_returns.index, [ticker])

    print("Loading sentiment model...")
    # Setting device to -1 uses CPU, 0 uses GPU. CPU is safer.
    sentiment_pipeline = pipeline("sentiment-analysis", device=-1)

    print("Scoring headlines (mock LLM inference)...")
    sentiments = []
    for headline in news_df["Headline"]:
        result = sentiment_pipeline(headline)[0]
        score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        sentiments.append(score)

    news_df["Sentiment_Score"] = sentiments
    daily_sentiment = news_df.groupby("Date")["Sentiment_Score"].mean()
    daily_sentiment = daily_sentiment.reindex(stock_returns.index).fillna(0)

    # Generate signal
    signal = pd.Series(0, index=daily_sentiment.index, dtype=float)
    signal[daily_sentiment > 0.3] = 1.0  # Go Long if positive sentiment
    signal[daily_sentiment < -0.3] = -1.0  # Go Short if negative sentiment

    # Shift to trade T+1 open based on day T close/news
    signal = signal.shift(1).fillna(0)

    # Backtest
    strategy_returns = signal * stock_returns[ticker]
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_stock = (1 + stock_returns[ticker]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy.index, cum_strategy, label="Sentiment Strategy", color="blue")
    plt.plot(
        cum_stock.index,
        cum_stock,
        label=f"{ticker} Buy & Hold",
        color="gray",
        linestyle="--",
    )

    plt.title(f"AI Sentiment Trading Strategy ({ticker})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/sentiment_strategy.png")
    print("Saved plot to plots/sentiment_strategy.png")


if __name__ == "__main__":
    main()
