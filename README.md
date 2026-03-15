# Quant Mini-Projects

**About this repo:** A collection of quantitative finance projects including rolling averages, portfolio risk metrics, factor investing, backtesting, probability puzzles, and AI-driven sentiment trading. Includes Python scripts, Jupyter notebooks, and visualizations.

## Projects

| Project | Description |
|---|---|
| Rolling Average Calculator | Computes rolling averages of stock prices using a sliding window. |
| Risk Metrics Analysis | Calculates daily returns, rolling volatility, and maximum drawdown; visualizes cumulative returns. |
| Factor Investing Engine | Ranks stocks using momentum and volatility factors and selects portfolio holdings. |
| Backtester | Simulates trading strategies and calculates portfolio returns, cumulative returns, and Sharpe ratio. |
| Probability Puzzles | Solves expected value and probability problems (coin flips, dice, cards). |
| AI Sentiment Trading | Scores financial news headlines with a Transformer model, generates buy/sell signals, and backtests strategy. |

## Files & Structure

```
quant-mini-projects/
├── data/                  # stock prices, news headlines
├── plots/                 # generated charts
├── scripts/               # Python scripts for each project
├── notebooks/             # Probability puzzles
├── README.md
├── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

python scripts/rolling_average.py
python scripts/factor_engine.py

jupyter notebook notebooks/probability_puzzles.ipynb
```

## Results

* Rolling averages and backtests visualize stock trends.
* Factor engine evaluates multi-factor portfolio performance.
* Probability puzzles include expected value calculations.
* AI sentiment trading maps headline sentiment to trading signals.

## Future Work

* Multi-factor weighting and ML integration
* Additional alternative data sources
* Realistic transaction costs and slippage
* Multi-asset portfolio support
