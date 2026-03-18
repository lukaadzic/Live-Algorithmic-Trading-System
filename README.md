# Mini Quantitative Research Pipeline for Developing, Backtesting, and Paper-Trading Systematic Strategies in Python

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Implemented Modules](#implemented-modules)
- [Example Results](#example-results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This repository contains a mini quantitative research pipeline for developing, testing, and analyzing systematic trading strategies using Python.

It replicates key components of workflows commonly used by quantitative researchers in hedge funds and prop trading firms, including:

- Financial data collection
- Signal generation
- Strategy backtesting
- Portfolio construction
- Risk analysis
- Performance visualization
- Real-time AI sentiment-driven trading

The system can simulate trading strategies on historical data and also execute paper trading with real-time signals while logging AI-driven trades, portfolio positions, and PnL.

### Key Features
- Modular research pipeline for systematic trading experiments
- Historical market data ingestion via yfinance
- Moving-average trend-following strategies
- Factor-based portfolio allocation
- Backtesting with cumulative returns, Sharpe ratio, and equity curves
- Portfolio risk analysis including maximum drawdown
- Transformer-based financial sentiment trading prototype
- Hybrid live paper trading engine with AI overrides
- Live plotting of equity and positions with matplotlib.animation

## Objectives
- Understand the end-to-end workflow of quantitative trading research
- Implement systematic trading strategies
- Build a modular backtesting framework
- Analyze risk and performance metrics
- Explore factor-based portfolio construction and AI-driven signal generation
- Build a live paper trading simulation with both long and short positions

## Project Structure
```text
quant_mini_project/
├── data/                  # stock prices, news headlines
├── plots/                 # generated charts
├── scripts/               # Python scripts for each module
├── notebooks/             # Probability puzzles and research notebooks
├── README.md
└── requirements.txt
```

## Data Sources
Data is sourced from Yahoo Finance via the `yfinance` Python library.

Tracked assets in experiments:
- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Amazon.com Inc. (AMZN)
- NVIDIA Corporation (NVDA)

For AI-driven sentiment trading, financial news headlines are fetched dynamically and scored with a transformer model.

## Implemented Modules

### 1. Moving Average Strategy
Implements a basic trend-following strategy using rolling averages.
- **Features**: Price vs moving average signals, Long/flat trading logic, Visualizations of price and indicator
- **Script**: `scripts/rolling_average.py`
- **Output**: `plots/price_vs_ma.png`

### 2. Strategy Backtesting Engine
Simulates historical trading performance.
- **Metrics**: Cumulative returns, Sharpe ratio, Equity curve visualization
- **Script**: `scripts/backtester.py`
- **Output**: `plots/backtest_equity_curve.png`

### 3. Factor Portfolio Engine
Allocates capital across multiple assets using factor signals.
- **Script**: `scripts/factor_engine.py`
- **Output**: `plots/factor_portfolio.png`

### 4. Risk Metrics Analysis
Calculates portfolio risk statistics:
- **Metrics**: Cumulative returns, Maximum drawdown, Risk profile over time
- **Script**: `scripts/risk_metrics.py`
- **Output**: `plots/cumulative_vs_drawdown.png`

### 5. AI Sentiment Trading Prototype
Prototype combining news sentiment analysis with trading signals.
- **Features**: Transformer-based sentiment scoring for financial headlines, Generation of daily signals (BULLISH, BEARISH, NEUTRAL), Extraction of top headlines driving the signal
- **Script**: `scripts/real_ai_sentiment.py`
- **Output**: Dataframe of sentiment signals and top headlines

### 6. Paper Trading / Live Signal Execution
Live simulation with long and short execution, integrating AI sentiment signals.
- **Features**: Real-time Alpaca Paper Trading API integration, AI sentiment overrides technical strategies, Automatic logging of top headlines for trades, Position size tracking for long/short reversals, Live equity tracking
- **Script**: `scripts/live_trading.py`
- **Output**: Console logs and `paper_trading.log` with PnL

### 7. Live Equity Visualization
Real-time equity curve plotting using `matplotlib.animation`.
- **Features**: Reads paper trading CSV every 10 seconds, Updates portfolio equity, PnL, and positions dynamically, Works alongside `live_trading.py` for cockpit-like monitoring
- **Script**: `scripts/live_chart.py`
- **Output**: Live matplotlib window

## Example Results

Example AI-driven trades during a live paper trading run:
- **AAPL**: BULLISH → Long 1 share
- **MSFT, AMZN, NVDA**: BEARISH → Short 1 share each
- **Global PnL**: +$0.76 within first few ticks

The system correctly tracked long and short positions, logged top headlines, and reflected market moves in equity dynamically.

## Installation

Clone the repository:
```bash
git clone https://github.com/markopom/quant_mini_project.git
cd quant_mini_project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run strategy modules
python scripts/rolling_average.py
python scripts/factor_engine.py
python scripts/backtester.py

# Run AI sentiment trading prototype
python scripts/real_ai_sentiment.py

# Run live paper trading (requires Alpaca API keys in .env)
python scripts/live_trading.py

# Open a live equity chart dashboard
python scripts/live_chart.py

# Explore probability puzzles
jupyter notebook notebooks/probability_puzzles.ipynb
```

## Technologies Used
- **Python 3.10**
- **pandas, numpy, matplotlib**
- **yfinance**
- **transformers** (Hugging Face)
- **Alpaca Paper Trading API**
- Logging, CSV tracking, and live plotting

## Future Improvements
### Realistic Trading
- Incorporate transaction costs, slippage, and bid-ask spreads
- Extend risk modeling to leverage and margin

### Advanced Validation
- Walk-forward testing for strategy robustness
- Benchmark comparisons (buy-and-hold, equal-weighted portfolio, market index)

### AI & Quant Enhancements
- Multi-factor portfolio models
- Reinforcement learning strategies
- AI-sentiment driven pair trading and hedging
- Volatility-targeted position sizing

## Educational Purpose
This project is a learning exercise demonstrating:
- Transforming financial data into trading signals
- Evaluating systematic strategies
- Integrating AI sentiment for live paper trading
- Risk-aware long and short execution

## License
MIT License
