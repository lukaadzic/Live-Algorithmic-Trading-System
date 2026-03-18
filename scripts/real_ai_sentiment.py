"""
AI Sentiment Assessment Module

This module connects with external financial APIs to parse headlines, assign
NLP confidence scores via FinBERT, and generate market-moving sentiment overrides.
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from transformers import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.config as cfg

logging.basicConfig(level=logging.INFO)


class AISentimentEngine:
    """
    Multi-API AI sentiment engine with market-moving filtering.
    It consolidates APIs, tests sentiment through an LLM sequence, and caches weight decays.
    """

    def __init__(self, decay_lambda: float = 0.1):
        # Load keys from configuration module
        self.newsapi_key = cfg.NEWS_API_KEY if cfg.NEWS_API_KEY else None
        self.newsdata_key = cfg.NEWSDATA_API_KEY if cfg.NEWSDATA_API_KEY else None
        self.finnhub_key = cfg.FINNHUB_API_KEY if cfg.FINNHUB_API_KEY else None
        self.llm_api_key = cfg.LLM_API_KEY if cfg.LLM_API_KEY else None

        self.decay_lambda = decay_lambda

        logging.info("Loading FinBERT sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert", device=-1
        )

        self.keywords: List[str] = [
            "earnings",
            "revenue",
            "profit",
            "guidance",
            "acquisition",
            "merger",
            "buyout",
            "CEO",
            "CFO",
            "resigns",
            "fired",
            "lawsuit",
            "investigation",
            "regulation",
            "upgrade",
            "downgrade",
            "rating",
            "forecast",
            "outlook",
        ]

    # -----------------------------
    # API fetch functions
    # -----------------------------
    def fetch_news_newsapi(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch targeted news using the standard NewsAPI gateway."""
        if not self.newsapi_key:
            return pd.DataFrame()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "apiKey": self.newsapi_key,
            "pageSize": 100,
        }
        try:
            r = requests.get(url, params=params).json()
            articles = r.get("articles", [])
            if not isinstance(articles, list):
                logging.warning(f"Unexpected results format from NewsAPI: {r}")
                return pd.DataFrame()

            rows = []
            for a in articles:
                if isinstance(a, dict) and "title" in a:
                    dt = (
                        pd.to_datetime(a.get("publishedAt"))
                        if a.get("publishedAt")
                        else datetime.now()
                    )
                    rows.append(
                        {
                            "Date": dt,
                            "Headline": a["title"],
                            "Source": a.get("source", {}).get("name", "newsapi"),
                        }
                    )
            return pd.DataFrame(rows)
        except Exception as e:
            logging.error(f"NewsAPI fetch error: {e}")
            return pd.DataFrame()

    def fetch_news_newsdata(self, ticker: str) -> pd.DataFrame:
        """Fetch using the NewsData.io external provider."""
        if not self.newsdata_key:
            return pd.DataFrame()

        url = "https://newsdata.io/api/1/news"
        params = {"apikey": self.newsdata_key, "q": ticker, "language": "en"}
        try:
            r = requests.get(url, params=params).json()
            results = r.get("results", [])

            if not isinstance(results, list):
                logging.warning(f"Unexpected results format from NewsData.io: {r}")
                return pd.DataFrame()

            rows = []
            for a in results:
                if isinstance(a, dict) and "title" in a:
                    pub_date = a.get("pubDate")
                    dt = pd.to_datetime(pub_date) if pub_date else datetime.now()
                    rows.append(
                        {
                            "Date": dt,
                            "Headline": a["title"],
                            "Source": a.get("source_id", "newsdata"),
                        }
                    )
            return pd.DataFrame(rows)

        except Exception as e:
            logging.error(f"NewsData.io fetch error: {e}")
            return pd.DataFrame()

    def fetch_news_finnhub(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fetch targeted company news from the Finnhub portal."""
        if not self.finnhub_key:
            return pd.DataFrame()

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_key,
        }
        try:
            r = requests.get(url, params=params).json()
            if not isinstance(r, list):
                logging.warning(f"Unexpected results format from Finnhub: {r}")
                return pd.DataFrame()

            rows = []
            for a in r:
                if isinstance(a, dict) and "headline" in a:
                    dt = pd.to_datetime(
                        a.get("datetime", int(datetime.now().timestamp())), unit="s"
                    )
                    rows.append(
                        {
                            "Date": dt,
                            "Headline": a["headline"],
                            "Source": a.get("source", "finnhub"),
                        }
                    )
            return pd.DataFrame(rows)

        except Exception as e:
            logging.error(f"Finnhub fetch error: {e}")
            return pd.DataFrame()

    # -----------------------------
    # Multi-source fetch
    # -----------------------------
    def fetch_all_news(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Combines multiple APIs and returns a single concatenated DataFrame."""
        dfs = []

        logging.info(f"Fetching news for {ticker}...")

        df_newsapi = self.fetch_news_newsapi(ticker, start_date, end_date)
        if not df_newsapi.empty:
            dfs.append(df_newsapi)

        df_newsdata = self.fetch_news_newsdata(ticker)
        if not df_newsdata.empty:
            dfs.append(df_newsdata)

        df_finnhub = self.fetch_news_finnhub(ticker, start_date, end_date)
        if not df_finnhub.empty:
            dfs.append(df_finnhub)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["Headline"]).sort_values("Date")
            logging.info(f"Fetched total {len(df)} unique articles.")
            return df
        else:
            logging.warning("No valid news data could be fetched.")
            return pd.DataFrame()

    # -----------------------------
    # Market-moving filter
    # -----------------------------
    def filter_market_moving(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops noise and retains highly impactful headlines."""
        if df.empty:
            return df
        lower_headlines = df["Headline"].str.lower()
        mask = lower_headlines.apply(
            lambda x: any(k.lower() in x for k in self.keywords)
        )
        df_filtered = df[mask].copy()

        logging.info(
            f"Filtered {len(df_filtered)} market-moving articles out of {len(df)}"
        )
        return df_filtered

    # -----------------------------
    # Sentiment scoring
    # -----------------------------
    def score_sentiment(self, headlines: List[str]) -> List[float]:
        """Utilizes the HuggingFace transformer to score NLP outcomes."""
        if self.llm_api_key:
            logging.info("LLM API key found: using LLM (placeholder)")
            return [0.0 for _ in headlines]
        else:
            scores = []
            for h in headlines:
                try:
                    r = self.sentiment_pipeline(h)[0]
                    score = (
                        r["score"] if r["label"].lower() == "positive" else -r["score"]
                    )
                    scores.append(score)
                except:
                    scores.append(0.0)
            return scores

    # -----------------------------
    # Time decay
    # -----------------------------
    def apply_time_decay(
        self, df: pd.DataFrame, now: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Diminishes impact of old news events using exponential decay."""
        if df.empty:
            return df
        now_dt = now or datetime.now()

        # Strip timezone from datetimes if present
        if df["Date"].dt.tz is not None:
            now_for_diff = datetime.now(timezone.utc).replace(tzinfo=None)
            dates_for_diff = df["Date"].dt.tz_localize(None)
        else:
            now_for_diff = now_dt
            dates_for_diff = df["Date"]

        hours = (now_for_diff - dates_for_diff).dt.total_seconds() / 3600
        hours = np.maximum(hours, 0)

        df["Weight"] = np.exp(-self.decay_lambda * hours)
        df["Weighted_Sentiment"] = df["Sentiment_Score"] * df["Weight"]
        return df

    # -----------------------------
    # Aggregation and signal generation
    # -----------------------------
    def aggregate(self, df: pd.DataFrame) -> pd.Series:
        """Groups outputs uniquely per active trading day."""
        if df.empty:
            return pd.Series(dtype=float)
        df["DateOnly"] = df["Date"].dt.date
        try:
            return df.groupby("DateOnly").apply(
                lambda x: x["Weighted_Sentiment"].sum() / (x["Weight"].sum() + 1e-8),
                include_groups=False,
            )
        except TypeError:
            return df.groupby("DateOnly").apply(
                lambda x: x["Weighted_Sentiment"].sum() / (x["Weight"].sum() + 1e-8)
            )

    def generate_signal(self, sentiment: pd.Series) -> pd.Series:
        """Threshold-based integer binning for algorithmic consumption."""
        signal = pd.Series(0.0, index=sentiment.index)
        signal[sentiment > 0.2] = 1
        signal[sentiment < -0.2] = -1
        return signal

    # -----------------------------
    # Full pipeline
    # -----------------------------
    def run(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Invokes the full execution pipeline end-to-end."""
        df = self.fetch_all_news(ticker, start_date, end_date)
        df = self.filter_market_moving(df)
        if df.empty:
            logging.warning(f"No relevant news found for {ticker}")
            return pd.DataFrame(), None

        df["Sentiment_Score"] = self.score_sentiment(df["Headline"].tolist())
        df = self.apply_time_decay(df)
        sentiment = self.aggregate(df)
        signal = self.generate_signal(sentiment)

        # Find top headline (highest absolute weighted sentiment)
        top_headline = (
            df.loc[df["Weighted_Sentiment"].abs().idxmax(), "Headline"]
            if not df.empty
            else None
        )

        return pd.DataFrame({"sentiment": sentiment, "signal": signal}), top_headline


if __name__ == "__main__":
    engine = AISentimentEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    result_df, top_event = engine.run("AAPL", start_date, end_date)
    print("\n--- Final Sentiment Signal ---")
    if result_df is not None and not result_df.empty:
        print(result_df.tail())
        print(f"\nTop Headline: {top_event}")
    else:
        print("No signal generated.")
