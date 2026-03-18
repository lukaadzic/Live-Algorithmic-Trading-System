"""
Live Trading Execution Module

This script runs the paper trading engine, merging a rolling average technical
strategy with an AI-driven short-selling sentiment override system.

It logs the top headlines and manages active stop-loss/take-profit parameters.
"""

import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.config as cfg
from scripts.real_ai_sentiment import AISentimentEngine

# Configure logging to file and console for better tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(cfg.LOG_FILE)],
)


def init_csv() -> None:
    """Initialize CSV with headers if it doesn't exist."""
    if not os.path.exists(cfg.CSV_FILE):
        with open(cfg.CSV_FILE, mode="w", newline="") as file:
            fieldnames = ["timestamp", "total_equity", "global_unrealized_pnl"] + [
                f"{sym}_pnl" for sym in cfg.SYMBOLS
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()


def execute_order(
    client: TradingClient, symbol: str, qty: float, side: OrderSide
) -> None:
    """Helper to execute a market order and catch exceptions."""
    try:
        order_data = MarketOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
        )
        client.submit_order(order_data=order_data)
        logging.info(f"Market order submitted successfully: {side.name} {qty} {symbol}")
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")


def main() -> None:
    """Main execution loop for the paper trading engine."""
    logging.info("Starting Multi-Asset Paper Trading Simulator (Hybrid AI Mode)...")
    if (
        not cfg.ALPACA_API_KEY
        or not cfg.ALPACA_SECRET_KEY
        or cfg.ALPACA_API_KEY == "your_real_key"
    ):
        logging.error("Alpaca API credentials not found or not configured.")
        return

    init_csv()

    trading_client = TradingClient(
        cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, paper=True
    )
    data_client = StockHistoricalDataClient(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY)

    sentiment_engine = AISentimentEngine()
    last_sentiment_update: Dict[str, datetime] = {}
    cached_sentiment: Dict[str, int] = {}
    top_headlines: Dict[str, Any] = {}

    price_history: Dict[str, list] = defaultdict(list)

    try:
        logging.info("Fetching initial positions and account data...")
        account = trading_client.get_account()
        logging.info(
            f"Initial Account Equity: ${account.equity} | Buying Power: ${account.buying_power}"
        )

        while True:
            logging.info("-" * 60)

            # Fetch latest data
            request_params = StockLatestQuoteRequest(symbol_or_symbols=cfg.SYMBOLS)
            try:
                latest_quotes = data_client.get_stock_latest_quote(request_params)
            except Exception as e:
                logging.error(f"Error fetching quotes: {e}")
                time.sleep(cfg.TICK_INTERVAL)
                continue

            # Update account state
            try:
                positions = trading_client.get_all_positions()
                account = trading_client.get_account()
                position_map = {pos.symbol: pos for pos in positions}

                global_unrealized_pl = sum(
                    float(pos.unrealized_pl) for pos in positions
                )
                logging.info(
                    f"Current Equity: ${account.equity} | Global PnL: ${global_unrealized_pl:.2f}"
                )

            except Exception as e:
                logging.error(f"Error fetching account/positions: {e}")
                time.sleep(cfg.TICK_INTERVAL)
                continue

            csv_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_equity": account.equity,
                "global_unrealized_pnl": f"{global_unrealized_pl:.2f}",
            }

            for sym in cfg.SYMBOLS:
                if sym not in latest_quotes:
                    logging.warning(f"No quote data returned for {sym}")
                    csv_row[f"{sym}_pnl"] = "0.00"
                    continue

                quote_data = latest_quotes[sym]
                price = (float(quote_data.ask_price) + float(quote_data.bid_price)) / 2

                price_history[sym].append(price)
                if len(price_history[sym]) > cfg.MA_WINDOW:
                    price_history[sym].pop(0)

                current_qty = 0.0
                unrealized_pl = 0.0
                if sym in position_map:
                    pos = position_map[sym]
                    current_qty = float(pos.qty)
                    unrealized_pl = float(pos.unrealized_pl)

                csv_row[f"{sym}_pnl"] = f"{unrealized_pl:.2f}"

                now = datetime.now()
                # Run AI signal generation respecting cache intervals
                if sym not in last_sentiment_update or (
                    now - last_sentiment_update[sym]
                ) > timedelta(hours=cfg.SENTIMENT_CACHE_HOURS):
                    logging.info(f"Refetching robust AI sentiment for {sym}...")
                    start_date = now - timedelta(days=7)
                    sentiment_df, top_headline = sentiment_engine.run(
                        sym, start_date, now
                    )

                    if sentiment_df is not None and not sentiment_df.empty:
                        cached_sentiment[sym] = int(sentiment_df["signal"].iloc[-1])
                        top_headlines[sym] = top_headline
                    else:
                        cached_sentiment[sym] = 0
                        top_headlines[sym] = None

                    last_sentiment_update[sym] = now

                ai_signal = int(cached_sentiment.get(sym, 0))
                ai_display = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(
                    ai_signal, "UNKNOWN"
                )

                if len(price_history[sym]) == cfg.MA_WINDOW:
                    rolling_avg = sum(price_history[sym]) / cfg.MA_WINDOW

                    pos_str = (
                        f" | Pos: {int(current_qty)} (PnL: ${unrealized_pl:.2f})"
                        if current_qty != 0
                        else ""
                    )
                    logging.info(
                        f"{sym:<5} | Mid: ${price:<8.2f} | MA: ${rolling_avg:<8.2f} | AI: {ai_display}{pos_str}"
                    )

                    # 1. RISK MANAGEMENT LAYER
                    risk_triggered = False
                    if current_qty != 0:
                        pos_data = position_map[sym]
                        entry_price = float(pos_data.avg_entry_price)

                        # Handle reverse PnL if shorting
                        if current_qty > 0:
                            pnl_pct = (price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - price) / entry_price

                        if pnl_pct <= -cfg.STOP_LOSS_PCT:
                            logging.warning(
                                f"[STOP LOSS] triggered for {sym} ({pnl_pct:.2%})"
                            )
                            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
                            execute_order(trading_client, sym, abs(current_qty), side)
                            risk_triggered = True

                        elif pnl_pct >= cfg.TAKE_PROFIT_PCT:
                            logging.info(
                                f"[TAKE PROFIT] triggered for {sym} ({pnl_pct:.2%})"
                            )
                            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
                            execute_order(trading_client, sym, abs(current_qty), side)
                            risk_triggered = True

                    if risk_triggered:
                        continue

                    # 2. AI SENTIMENT OVERRIDE
                    if ai_signal == 1 and current_qty <= 0:
                        logging.info(f"[AI BUY OVERRIDE] Signal for {sym}")
                        if top_headlines.get(sym):
                            logging.info(
                                f"[HEADLINE] Triggering event: '{top_headlines[sym]}'"
                            )

                        qty_to_buy = (
                            abs(current_qty) + cfg.TRADE_QTY
                            if current_qty < 0
                            else cfg.TRADE_QTY
                        )
                        execute_order(trading_client, sym, qty_to_buy, OrderSide.BUY)
                        continue

                    elif ai_signal == -1 and current_qty >= 0:
                        logging.info(f"[AI SELL OVERRIDE] Signal for {sym}")
                        if top_headlines.get(sym):
                            logging.info(
                                f"[HEADLINE] Triggering event: '{top_headlines[sym]}'"
                            )

                        qty_to_sell = (
                            current_qty + cfg.TRADE_QTY
                            if current_qty > 0
                            else cfg.TRADE_QTY
                        )
                        execute_order(trading_client, sym, qty_to_sell, OrderSide.SELL)
                        continue

                    # 3. ROLLING AVERAGE FALLBACK
                    if ai_signal == 0:
                        if price > rolling_avg * 1.0005 and current_qty <= 0:
                            logging.info(
                                f"[MA BUY SIGNAL] triggered for {sym} (Price > MA)"
                            )
                            qty_to_buy = (
                                abs(current_qty) + cfg.TRADE_QTY
                                if current_qty < 0
                                else cfg.TRADE_QTY
                            )
                            execute_order(
                                trading_client, sym, qty_to_buy, OrderSide.BUY
                            )

                        elif price < rolling_avg * 0.9995 and current_qty >= 0:
                            logging.info(
                                f"[MA SELL SIGNAL] triggered for {sym} (Price < MA)"
                            )
                            qty_to_sell = (
                                current_qty + cfg.TRADE_QTY
                                if current_qty > 0
                                else cfg.TRADE_QTY
                            )
                            execute_order(
                                trading_client, sym, qty_to_sell, OrderSide.SELL
                            )
                else:
                    warm_up = len(price_history[sym])
                    logging.info(
                        f"{sym:<5} | Mid: ${price:<8.2f} | Warming up MA ({warm_up}/{cfg.MA_WINDOW})"
                    )

            with open(cfg.CSV_FILE, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=csv_row.keys())
                writer.writerow(csv_row)

            logging.info(
                f"Sleeping for {cfg.TICK_INTERVAL} seconds before next tick...\n"
            )
            time.sleep(cfg.TICK_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Paper trading simulator manually stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error in trading loop: {e}")


if __name__ == "__main__":
    main()
