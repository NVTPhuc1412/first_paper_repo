"""
data_scraper.py
---------------
Downloads raw OHLCV data from Yahoo Finance for all tickers defined in
Tickers, validates minimum length, and writes one CSV per ticker.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from tickers import Tickers

__all__ = ["fetch_ticker", "is_sufficient", "run_scraper"]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

START_DATE    = "2014-01-01"   # extra year for rolling-window warm-up
END_DATE      = "2025-12-31"
OUTPUT_DIR    = Path("../../data/raw_tickers_data")
API_SLEEP_SEC = 0.5            # be respectful with the yfinance API

REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV history for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        Cleaned DataFrame with REQUIRED_COLS, or None on failure.
    """
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end)
    except Exception:
        logger.exception("Network error fetching %s", ticker)
        return None

    if hist.empty:
        return None

    hist = hist.reset_index()
    date_col = hist.columns[0]
    if date_col != "Date":
        hist = hist.rename(columns={date_col: "Date"})

    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date

    available = [c for c in REQUIRED_COLS if c in hist.columns]
    if len(available) < len(REQUIRED_COLS):
        missing = set(REQUIRED_COLS) - set(available)
        logger.warning("  ✗ %s: missing columns %s", ticker, missing)
        return None

    return hist[REQUIRED_COLS]


def is_sufficient(df: pd.DataFrame, min_rows: int) -> bool:
    """Check whether a DataFrame has more than *min_rows* rows."""
    return len(df) > min_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def run_scraper(output_dir: Path | None = None) -> None:
    """Fetch and save OHLCV data for all tickers in the registry.

    Args:
        output_dir: Directory to write CSVs. Defaults to OUTPUT_DIR.
    """
    tickers = Tickers()
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Minimum row threshold: n_samples + 200 warm-up rows
    min_rows = tickers.n_samples_per_ticker + 200

    fetch_set = tickers.all_fetch_tickers
    logger.info("Scraping %d tickers  [%s → %s]", len(fetch_set), START_DATE, END_DATE)

    successful, failed, total_rows = 0, 0, 0

    for ticker in sorted(fetch_set):
        df = fetch_ticker(ticker, START_DATE, END_DATE)

        if df is None:
            logger.warning("  ✗ %s: no data returned", ticker)
            failed += 1

        elif not is_sufficient(df, min_rows):
            logger.warning("  ✗ %s: only %d rows (need > %d)", ticker, len(df), min_rows)
            failed += 1

        else:
            df.to_csv(out / f"{ticker}.csv", index=False)
            total_rows += len(df)
            successful += 1
            logger.info("  ✓ %s: %d rows saved", ticker, len(df))

        time.sleep(API_SLEEP_SEC)

    logger.info(
        "Done — %d succeeded, %d failed, %d total rows",
        successful, failed, total_rows,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_scraper()