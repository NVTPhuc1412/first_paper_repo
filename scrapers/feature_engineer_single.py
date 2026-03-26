"""
feature_engineer_single.py
--------------------------
Single-ticker feature engineering for the analysis pipeline.
Wraps the existing prepare_df() with scaler fitting/transform.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add the stock_data_scraper directory so we can import prepare_df
_SCRAPER_DIR = Path(__file__).resolve().parent / "stock_data_scraper"
if str(_SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRAPER_DIR))

from feature_engineer import prepare_df  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["engineer_features"]


def engineer_features(
    ticker_df: pd.DataFrame,
    market_df: pd.DataFrame,
    n_samples: int = 2765,
    train_split: float = 0.8,
    save_dir: Path | None = None,
    scaler_dir: Path | None = None,
    ticker_name: str = "",
) -> tuple[pd.DataFrame, RobustScaler, list[str]]:
    """Compute engineered features and scale for a single ticker.

    Args:
        ticker_df:   Raw OHLCV DataFrame for the ticker.
        market_df:   Raw OHLCV DataFrame for the market benchmark.
        n_samples:   Number of rows to keep (tail).
        train_split: Fraction used for scaler fitting.
        save_dir:    If given, save the scaled CSV here.
        scaler_dir:  If given, save the fitted scaler here.
        ticker_name: Ticker symbol (for file naming).

    Returns:
        (scaled_df, scaler, feature_names)
    """
    logger.info("Engineering features for %s...", ticker_name)

    # Compute engineered features (exclude raw price columns)
    df_eng = prepare_df(ticker_df, market_df, exclude_price=True)

    # Keep the last n_samples rows
    if len(df_eng) > n_samples:
        df_eng = df_eng.iloc[-n_samples:].reset_index(drop=True)

    feature_names = list(df_eng.columns)
    logger.info("  Features (%d): %s", len(feature_names), feature_names)

    # Fit scaler on training portion
    train_end = int(len(df_eng) * train_split)
    scaler = RobustScaler()
    scaler.fit(df_eng.iloc[:train_end])

    # Transform full dataset
    scaled = pd.DataFrame(scaler.transform(df_eng), columns=feature_names)
    logger.info("  Scaled %d rows (train: %d, test: %d)",
                len(scaled), train_end, len(scaled) - train_end)

    # Save outputs
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        scaled.to_csv(save_dir / f"{ticker_name}.csv", index=False)
        logger.info("  Saved scaled CSV → %s", save_dir / f"{ticker_name}.csv")

    if scaler_dir:
        scaler_dir = Path(scaler_dir)
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_file = scaler_dir / f"{ticker_name}_scaler.joblib"
        joblib.dump(scaler, scaler_file, compress=3)
        logger.info("  Saved scaler → %s", scaler_file)

    return scaled, scaler, feature_names
