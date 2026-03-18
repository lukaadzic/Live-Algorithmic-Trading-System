import os

from dotenv import load_dotenv

load_dotenv()

# API Configuration
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")

NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
NEWSDATA_API_KEY: str = os.getenv("NEWSDATA_API_KEY", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

# Trading Parameters
SYMBOLS: list[str] = ["AAPL", "MSFT", "AMZN", "NVDA"]
TRADE_QTY: int = 1
MA_WINDOW: int = int(os.getenv("MA_WINDOW", "5"))
TICK_INTERVAL: int = int(os.getenv("TICK_INTERVAL", "60"))

# Risk Management
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))

# AI Sentiment Rate Limit Cache
SENTIMENT_CACHE_HOURS: int = 1

CSV_FILE: str = "paper_trading_history.csv"
LOG_FILE: str = "paper_trading.log"
