import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Define path to the CSV file outputted by the live_trading engine
CSV_FILE = "paper_trading_history.csv"


def animate(i):
    if not os.path.exists(CSV_FILE):
        return

    try:
        # Read the latest CSV stats
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Clear the plot and redraw the line so it looks like a live feed
        plt.clf()
        plt.plot(
            df["timestamp"],
            df["total_equity"],
            label="Total Equity",
            color="teal",
            linewidth=2,
        )
        plt.title("Live Paper Trading Equity Monitor", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Equity (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

    except Exception as e:
        # Silently pass file locks
        pass


if __name__ == "__main__":
    print("Starting Live Chart CSV Monitor. Keep this open to watch equity locally!")
    fig = plt.figure(figsize=(10, 5))

    # Reload the CSV every 10 seconds (10000ms)
    ani = FuncAnimation(fig, animate, interval=10000, cache_frame_data=False)

    plt.show()
