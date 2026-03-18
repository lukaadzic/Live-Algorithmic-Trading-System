import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SimpleBacktester:
    def __init__(self, prices, target_weights, transaction_cost=0.001):
        """
        prices: DataFrame of asset prices
        target_weights: Targets computed at close of day T
        transaction_cost: proportional cost (e.g., 0.001 for 10bps)
        """
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.target_weights = target_weights
        self.tc = transaction_cost

    def run(self):
        common_dates = self.returns.index.intersection(self.target_weights.index)
        returns = self.returns.loc[common_dates]
        target_weights = self.target_weights.loc[common_dates]

        # Shift weights to execute at day T+1 open (approx day T+1 return)
        weights = target_weights.shift(1).dropna()
        returns = returns.loc[weights.index]

        gross_returns = (weights * returns).sum(axis=1)

        weight_changes = weights.diff().abs()
        weight_changes.iloc[0] = weights.iloc[0].abs()

        costs = weight_changes.sum(axis=1) * self.tc
        net_returns = gross_returns - costs

        self.results = pd.DataFrame(
            {
                "Gross_Return": gross_returns,
                "Net_Return": net_returns,
                "Cumulative_Gross": (1 + gross_returns).cumprod(),
                "Cumulative_Net": (1 + net_returns).cumprod(),
            }
        )

        sharpe = (
            (net_returns.mean() / net_returns.std()) * np.sqrt(252)
            if net_returns.std() > 0
            else 0
        )
        print(f"Backtest Completed. Sharpe Ratio (Net): {sharpe:.2f}")
        return self.results

    def plot(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.results.index,
            self.results["Cumulative_Gross"],
            label="Gross Returns",
            color="green",
        )
        plt.plot(
            self.results.index,
            self.results["Cumulative_Net"],
            label="Net Returns",
            color="blue",
        )
        plt.title("Backtest Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")


def main():
    os.makedirs("plots", exist_ok=True)

    if not os.path.exists("data/historical_prices.csv"):
        print("Data not found. Run rolling_average.py first.")
        return

    prices = pd.read_csv("data/historical_prices.csv", index_col=0, parse_dates=True)

    # Signal: fast VS slow MA
    ma20 = prices.rolling(20).mean()
    ma50 = prices.rolling(50).mean()

    # 1 if MA20 > MA50 else 0
    signal = (ma20 > ma50).astype(float)

    # Equal weight allocation among assets with positive signal
    row_sums = signal.sum(axis=1)
    weights = signal.div(
        row_sums.replace(0, 1), axis=0
    )  # replace 0 to avoid division by zero
    weights[row_sums == 0] = 0
    weights = weights.fillna(0)

    tester = SimpleBacktester(prices, weights, transaction_cost=0.002)
    results = tester.run()

    results.to_csv("data/backtest_results.csv")
    tester.plot("plots/backtest_equity_curve.png")


if __name__ == "__main__":
    main()
