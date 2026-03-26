"""
stationary_check.py
--------------------
Filters training tickers to those with stationary log-returns, then ranks
survivors by a composite stability score (Hurst + vol CV + max Z).

Outputs a ranked DataFrame and prints the top-25 most stable tickers.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from tickers import Tickers

__all__ = ["log_returns", "is_stationary", "compute_stability", "run_stationarity_check"]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path("../../data/raw_tickers_data")
TOP_N        = 25
MIN_PRICES   = 500   # minimum Close observations needed for stability metrics
MIN_HURST    = 102   # minimum observations for lag-100 Hurst estimation

# Stability score weights (lower score = more stable)
W_HURST = 0.4
W_VOL   = 0.3
W_Z     = 0.3


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StabilityMetrics:
    """Per-ticker stability assessment."""
    ticker:    str
    hurst:     float
    vol_cv:    float
    max_z:     float
    score:     float


# ── Core functions ────────────────────────────────────────────────────────────

def log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute log-returns from the 'Close' column."""
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def is_stationary(returns: pd.Series) -> tuple[bool, float, float]:
    """ADF + KPSS dual test for stationarity.

    Stationary when ADF rejects unit-root (p < 0.05) AND KPSS fails to
    reject stationarity (p > 0.05).

    Returns:
        (stationary, p_adf, p_kpss)
    """
    p_adf = adfuller(returns)[1]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        try:
            _, p_kpss, _, _ = kpss(returns, regression="c", nlags="auto")
        except Exception:
            p_kpss = 0.0

    return (p_adf < 0.05 and p_kpss > 0.05), p_adf, p_kpss


def _hurst_exponent(prices: pd.Series) -> float:
    """Estimate Hurst exponent via log-log regression of τ vs lag.

    Returns NaN if the series is too short (< MIN_HURST observations).
    """
    if len(prices) < MIN_HURST:
        return float("nan")

    log_p = np.log(prices.values)
    lags  = range(2, 100)
    tau   = [np.std(log_p[lag:] - log_p[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])


def compute_stability(ticker: str, df: pd.DataFrame) -> StabilityMetrics | None:
    """Compute stability metrics for a single ticker.

    Returns a StabilityMetrics instance, or None if insufficient data.
    Lower score → more stable (better suited as a clean baseline).
    """
    prices = df["Close"].dropna()
    if len(prices) < MIN_PRICES:
        return None

    returns = np.log(prices / prices.shift(1)).dropna()

    hurst = _hurst_exponent(prices)
    if np.isnan(hurst):
        logger.warning("  ⚠ %s: too few observations for Hurst exponent", ticker)
        return None

    h_score = max(0.0, hurst)

    roll_vol = returns.rolling(window=21).std().dropna()
    vol_cv   = (roll_vol.std() / roll_vol.mean()) if roll_vol.mean() != 0 else 1.0

    max_z = float(np.abs(zscore(returns)).max())

    score = W_HURST * h_score + W_VOL * vol_cv + W_Z * (max_z / 10.0)

    return StabilityMetrics(
        ticker=ticker, hurst=hurst, vol_cv=vol_cv, max_z=max_z, score=score
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_stationarity_check(
    raw_data_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full stationarity + stability pipeline.

    Args:
        raw_data_dir: Directory containing ``{ticker}.csv`` files.

    Returns:
        DataFrame of tickers ranked by stability score (ascending).
    """
    data_dir = raw_data_dir or RAW_DATA_DIR
    tickers = Tickers()
    metrics: list[StabilityMetrics] = []
    skipped: list[str] = []

    for ticker in sorted(tickers.training_tickers):
        path = data_dir / f"{ticker}.csv"
        if not path.exists():
            skipped.append(ticker)
            continue

        df      = pd.read_csv(path)
        returns = log_returns(df)
        stationary, p_adf, p_kpss = is_stationary(returns)

        if not stationary:
            logger.info("  ✗ %s: non-stationary (ADF p=%.4f, KPSS p=%.4f)", ticker, p_adf, p_kpss)
            continue

        result = compute_stability(ticker, df)
        if result is not None:
            metrics.append(result)

    if skipped:
        logger.warning("Skipped (no CSV): %s", ", ".join(skipped))

    ranked = (
        pd.DataFrame([m.__dict__ for m in metrics])
        .sort_values("score")
        .reset_index(drop=True)
    )

    logger.info("Top %d most stable tickers:\n%s", TOP_N, ranked.head(TOP_N).to_string(index=True))
    return ranked


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_stationarity_check()