"""
fetch.py
--------
Single-ticker stock data fetching for the analysis pipeline.
Downloads OHLCV from Yahoo Finance for a ticker and the market benchmark.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV history for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start:  Start date YYYY-MM-DD.
        end:    End date YYYY-MM-DD.

    Returns:
        DataFrame with [Date, Open, High, Low, Close, Volume], or None.
    """
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
    except Exception:
        logger.exception("Network error fetching %s", ticker)
        return None

    if hist.empty:
        logger.warning("%s: no data returned", ticker)
        return None

    hist = hist.reset_index()
    date_col = hist.columns[0]
    if date_col != "Date":
        hist = hist.rename(columns={date_col: "Date"})

    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date

    available = [c for c in REQUIRED_COLS if c in hist.columns]
    if len(available) < len(REQUIRED_COLS):
        missing = set(REQUIRED_COLS) - set(available)
        logger.warning("%s: missing columns %s", ticker, missing)
        return None

    return hist[REQUIRED_COLS]


def fetch_stock_data(
    ticker: str,
    market_ticker: str = "SPY",
    start: str = "2014-01-01",
    end: str = "2025-12-31",
    min_rows: int = 2765,
    save_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch OHLCV data for a ticker and the market benchmark.

    Args:
        ticker:        Target ticker symbol.
        market_ticker: Market benchmark ticker (default SPY).
        start:         Start date.
        end:           End date.
        min_rows:      Minimum rows required.
        save_dir:      If given, save CSVs to this directory.

    Returns:
        (ticker_df, market_df) — both DataFrames with REQUIRED_COLS.

    Raises:
        ValueError: If either ticker fails to download or has too few rows.
    """
    logger.info("Fetching %s (%s → %s)...", ticker, start, end)
    ticker_df = fetch_ticker(ticker, start, end)
    if ticker_df is None or len(ticker_df) < min_rows:
        rows = len(ticker_df) if ticker_df is not None else 0
        raise ValueError(
            f"{ticker}: got {rows} rows, need >= {min_rows}. "
            f"Check ticker symbol or date range."
        )
    logger.info("  %s: %d rows", ticker, len(ticker_df))

    logger.info("Fetching market benchmark %s...", market_ticker)
    market_df = fetch_ticker(market_ticker, start, end)
    if market_df is None or len(market_df) < min_rows:
        rows = len(market_df) if market_df is not None else 0
        raise ValueError(
            f"{market_ticker}: got {rows} rows, need >= {min_rows}."
        )
    logger.info("  %s: %d rows", market_ticker, len(market_df))

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ticker_df.to_csv(save_dir / f"{ticker}.csv", index=False)
        market_df.to_csv(save_dir / f"{market_ticker}.csv", index=False)
        logger.info("  Saved raw CSVs → %s", save_dir)

    return ticker_df, market_df
