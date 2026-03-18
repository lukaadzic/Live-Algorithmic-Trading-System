import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def calculate_returns(prices):
    return prices.pct_change().dropna()


def calculate_rolling_volatility(returns, window=30):
    # Annualized rolling volatility (assuming 252 trading days)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return cumulative, drawdown


def main():
    os.makedirs("plots", exist_ok=True)

    if not os.path.exists("data/historical_prices.csv"):
        print("Data not found. Please run rolling_average.py first to fetch data.")
        return

    prices = load_data("data/historical_prices.csv")

    # Calculate daily returns for each asset, then create an equally-weighted portfolio
    asset_returns = calculate_returns(prices)
    portfolio_returns = asset_returns.mean(axis=1)

    volatility = calculate_rolling_volatility(portfolio_returns, window=30)
    cumulative_returns, drawdown = calculate_max_drawdown(portfolio_returns)

    max_dd = drawdown.min()
    print(f"Maximum Drawdown of EW Portfolio: {max_dd:.2%}")

    # Save metrics
    metrics = pd.DataFrame(
        {
            "Portfolio_Returns": portfolio_returns,
            "Rolling_Volatility_30d": volatility,
            "Cumulative_Returns": cumulative_returns,
            "Drawdown": drawdown,
        }
    )
    metrics.to_csv("data/risk_metrics.csv")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(
        cumulative_returns.index,
        cumulative_returns,
        label="Cumulative Returns",
        color="green",
    )
    ax1.set_title("Equal-Weighted Portfolio Cumulative Returns")
    ax1.set_ylabel("Growth Factor")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(drawdown.index, drawdown, label="Drawdown", color="red")
    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    ax2.set_title("Portfolio Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("plots/cumulative_vs_drawdown.png")
    print("Saved plot to plots/cumulative_vs_drawdown.png")


if __name__ == "__main__":
    main()
