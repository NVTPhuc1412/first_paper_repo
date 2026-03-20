"""
Synthetic OHLCV Anomaly Injection Pipeline
===========================================
Generates a mathematically sound, labeled ground truth dataset for validating
anomaly detection models. Injects Point, Contextual, and Collective anomalies
into mean-reverting OHLCV time series while preserving market microstructure.
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from tickers import Tickers

__all__ = [
    "run_pipeline",
    "generate_synthetic_ohlcv",
    "InjectionEvent",
    "TickerResult",
]

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
AnomalyType = Literal["Point", "Contextual", "Collective"]
Difficulty = Literal["Easy", "Medium", "Hard"]

DIFFICULTY_PARAMS = {
    "Point": {
        "Easy":   (6.0, 8.0),
        "Medium": (4.0, 6.0),
        "Hard":   (2.5, 4.0),
    },
    "Contextual": {
        "Easy":   (4.0, 5.0),
        "Medium": (2.5, 4.0),
        "Hard":   (1.5, 2.5),
    },
    "Collective": {
        "Easy":   (1.0, 1.5),
        "Medium": (0.5, 1.0),
        "Hard":   (0.2, 0.5),
    },
}

COLLECTIVE_DURATION = (8, 18)       # trading days
CONTEXTUAL_DURATION = (10, 20)      # trading days
POINT_FOOTPRINT     = 6             # T0 + 5 decay days for collision detection
COOLDOWN_DAYS       = 30            # minimum gap between events per ticker
START_BUFFER        = 63 + 252      # trading days excluded at the start
END_BUFFER          = 63            # trading days excluded at the end
TAPER_DAYS          = 3             # linear ramp-in / ramp-out length
MAX_RESAMPLE        = 100           # rejection-sampling attempt limit
TARGET_DENSITY      = 0.06          # fraction of rows that are anomalous
SIGMA_FLOOR         = 1e-5          # zero-noise clamp
PRICE_FLOOR         = 0.0001        # hard price floor


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class InjectionEvent:
    """Record of a single injected anomaly."""
    anomaly_type: AnomalyType
    difficulty:   Difficulty
    start_idx:    int
    end_idx:      int          # inclusive last affected index
    direction:    int          # +1 or -1
    magnitude:    float        # sampled sigma multiplier / variance multiplier


@dataclass
class TickerResult:
    """Container for one ticker's injection output."""
    ticker:   str
    df:       pd.DataFrame
    events:   list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1 — Pre-Processing & Immutable Baseline
# ---------------------------------------------------------------------------

def compute_immutable_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a 21-day rolling σ of log-returns on the RAW, unaltered data.
    Returns a frozen numpy array (same length as df) clamped at SIGMA_FLOOR.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rolling_sigma = log_ret.rolling(window=21, min_periods=1).std().values
    rolling_sigma = np.where(np.isnan(rolling_sigma), SIGMA_FLOOR, rolling_sigma)
    rolling_sigma = np.maximum(rolling_sigma, SIGMA_FLOOR)
    return rolling_sigma.copy()   # freeze — never modify this array


def get_eligible_range(df: pd.DataFrame) -> tuple[int, int]:
    """Return the (start, end) index range after applying the boundary buffer."""
    return START_BUFFER, len(df) - END_BUFFER - 1


# ---------------------------------------------------------------------------
# Phase 2 — Temporal Logic & Safety Boundaries
# ---------------------------------------------------------------------------

def build_taper_multipliers(window_len: int) -> np.ndarray:
    """
    Create a multiplier array of shape (window_len,) that linearly ramps
    from 0 → 1 over the first TAPER_DAYS, stays at 1, then ramps 1 → 0
    over the last TAPER_DAYS. Prevents seam artifacts.
    """
    mults = np.ones(window_len)
    ramp = min(TAPER_DAYS, window_len // 2)
    for i in range(ramp):
        mults[i]                  = (i + 1) / (ramp + 1)
        mults[window_len - 1 - i] = (i + 1) / (ramp + 1)
    return mults


def sample_injection_index(
    eligible_start: int,
    eligible_end:   int,
    occupied:       set[int],
    window_len:     int,
    rng:            np.random.Generator,
) -> Optional[int]:
    """Rejection-sample a start index whose window doesn't overlap occupied indices.

    Returns None if ``MAX_RESAMPLE`` attempts are exhausted.
    """
    for _ in range(MAX_RESAMPLE):
        idx = int(rng.integers(eligible_start, eligible_end - window_len + 2))
        blocked = any(
            (idx - COOLDOWN_DAYS) <= occ <= (idx + window_len - 1 + COOLDOWN_DAYS)
            for occ in occupied
        )
        if not blocked:
            return idx
    return None


def mark_occupied(occupied: set[int], start: int, end: int) -> None:
    """Mark indices [start, end] as occupied."""
    for i in range(start, end + 1):
        occupied.add(i)


# ---------------------------------------------------------------------------
# Phase 3 — Anomaly Archetypes
# ---------------------------------------------------------------------------

def inject_point_anomaly(
    df:    pd.DataFrame,
    sigma: np.ndarray,
    event: InjectionEvent,
    rng:   np.random.Generator,
) -> pd.DataFrame:
    """
    Single-day price spike/crash (T0) followed by a 5-day exponential
    volume decay. Price uses a log-return shock scaled by frozen σ.
    """
    df = df.copy()
    t0 = event.start_idx

    # --- Price shock at T0 ---
    shock = event.direction * event.magnitude * sigma[t0]
    factor = np.exp(shock)
    for col in ["Open", "High", "Low", "Close"]:
        df.at[t0, col] = df.at[t0, col] * factor

    # --- Volume: spike + 5-day exponential decay ---
    v0_mult = rng.uniform(3.0, 6.0)
    for t in range(6):                        # T0 through T5
        idx = t0 + t
        if idx >= len(df):
            break
        decay_mult = 1.0 + (v0_mult - 1.0) * np.exp(-0.5 * t)
        df.at[idx, "Volume"] = df.at[idx, "Volume"] * decay_mult

    return df


def inject_contextual_anomaly(
    df:    pd.DataFrame,
    sigma: np.ndarray,
    event: InjectionEvent,
    rng:   np.random.Generator,
) -> pd.DataFrame:
    """
    Volatility expansion: widens High-Low by a variance multiplier while
    preserving the directional sign of (Close - Open).
    """
    df = df.copy()
    start, end = event.start_idx, event.end_idx
    window_len = end - start + 1
    taper = build_taper_multipliers(window_len)

    for i, idx in enumerate(range(start, end + 1)):
        mult = event.magnitude * taper[i]

        o = df.at[idx, "Open"]
        c = df.at[idx, "Close"]
        h = df.at[idx, "High"]
        l = df.at[idx, "Low"]

        body = c - o
        mid  = (h + l) / 2.0
        half_range = (h - l) / 2.0 * mult

        new_h = mid + half_range
        new_l = mid - half_range

        # Preserve directional integrity (sign of body); handle doji
        if body != 0:
            scale = abs(body) * mult
            if body > 0:
                new_c = np.clip(o + scale, new_l, new_h)
                new_o = np.clip(o, new_l, new_c)
            else:
                new_c = np.clip(o - scale, new_l, new_h)
                new_o = np.clip(o, new_c, new_h)
        else:
            new_o, new_c = o, c   # doji preserved exactly

        df.at[idx, "Open"]  = new_o
        df.at[idx, "Close"] = new_c
        df.at[idx, "High"]  = new_h
        df.at[idx, "Low"]   = new_l

    return df


def inject_collective_anomaly(
    df:               pd.DataFrame,
    sigma:            np.ndarray,
    event:            InjectionEvent,
    rng:              np.random.Generator,
    shuffle_autocorr: bool = False,
) -> pd.DataFrame:
    """
    Persistent directional drift in log-return space.
    Gravity (window mean) is subtracted before adding drift.
    Optionally shuffles return order to destroy mean-reversion zigzag.
    """
    df = df.copy()
    start, end = event.start_idx, event.end_idx
    window_len = end - start + 1
    taper = build_taper_multipliers(window_len)

    closes = df["Close"].values.copy()
    # Slice: window_len returns covering [start, end], computed from
    # closes[start-1] through closes[end], yielding length = window_len.
    log_rets = np.log(closes[start:end + 2] / closes[start - 1:end + 1])

    # Centre returns (gravity suspension)
    window_rets = log_rets[:window_len].copy()
    window_rets -= window_rets.mean()

    if shuffle_autocorr:
        rng.shuffle(window_rets)

    # Add drift scaled by frozen σ and taper
    drift_per_day = event.direction * event.magnitude * sigma[start:end + 1]
    new_rets = window_rets + drift_per_day * taper

    # Reconstruct prices from new log-returns
    base_price = closes[start - 1]
    new_closes = base_price * np.exp(np.cumsum(new_rets))

    # Apply same multiplier to all OHLCV columns
    for i, idx in enumerate(range(start, end + 1)):
        ratio = new_closes[i] / df.at[idx, "Close"] if df.at[idx, "Close"] != 0 else 1.0
        for col in ["Open", "High", "Low", "Close"]:
            df.at[idx, col] = df.at[idx, col] * ratio

    return df


# ---------------------------------------------------------------------------
# Phase 4 — OHLCV Integrity Engine
# ---------------------------------------------------------------------------

def enforce_ohlcv_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce OHLCV constraints across the entire DataFrame.

    Rules applied:
      1. High ≥ max(Open, Close),  Low ≤ min(Open, Close)
      2. (handled during drift scaling)
      3. Open/Close clipped within [Low, High]
      4. All prices ≥ PRICE_FLOOR; Volume ≥ 0
    """
    df = df.copy()

    # Rule 4 first: clamp all prices above floor
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.maximum(df[col].values, PRICE_FLOOR)

    # Rule 3: clip Open & Close within High/Low
    df["Open"]  = np.clip(df["Open"].values,  df["Low"].values, df["High"].values)
    df["Close"] = np.clip(df["Close"].values, df["Low"].values, df["High"].values)

    # Rule 1: High ≥ max(Open, Close),  Low ≤ min(Open, Close)
    df["High"] = np.maximum(df["High"].values, np.maximum(df["Open"].values, df["Close"].values))
    df["Low"]  = np.minimum(df["Low"].values,  np.minimum(df["Open"].values, df["Close"].values))

    # Volumes must stay non-negative
    df["Volume"] = np.maximum(df["Volume"].values, 0.0)

    return df


# ---------------------------------------------------------------------------
# Phase 5 — Ground Truth Labels & Sanity Check
# ---------------------------------------------------------------------------

def add_ground_truth_columns(df: pd.DataFrame, events: list[InjectionEvent]) -> pd.DataFrame:
    df = df.copy()
    df["Is_Anomaly_Point"]       = False
    df["Is_Anomaly_Contextual"]  = False
    df["Is_Anomaly_Collective"]  = False

    for ev in events:
        col = f"Is_Anomaly_{ev.anomaly_type}"
        df.loc[ev.start_idx:ev.end_idx, col] = True

    df["anomaly"] = df[[
        "Is_Anomaly_Point",
        "Is_Anomaly_Contextual",
        "Is_Anomaly_Collective"
    ]].any(axis=1)

    return df


def sanity_check(df: pd.DataFrame, ticker: str) -> bool:
    """
    Assert logical OHLCV constraints hold for every row.
    Returns True if all checks pass; logs violations otherwise.
    """
    violations = []

    if (df["High"] < df["Open"] - 1e-8).any() or (df["High"] < df["Close"] - 1e-8).any():
        violations.append("High < Open or Close")
    if (df["Low"] > df["Open"] + 1e-8).any() or (df["Low"] > df["Close"] + 1e-8).any():
        violations.append("Low > Open or Close")
    if (df["High"] < df["Low"] - 1e-8).any():
        violations.append("High < Low")
    for col in ["Open", "High", "Low", "Close"]:
        if (df[col] < PRICE_FLOOR * 0.9).any():
            violations.append(f"{col} below price floor")
    if (df["Volume"] < 0).any():
        violations.append("Negative Volume")

    if violations:
        logger.warning("  [SANITY FAIL] %s: %s", ticker, ", ".join(violations))
        return False

    logger.info("  [SANITY OK]  %s: all constraints satisfied.", ticker)
    return True


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data:              dict[str, pd.DataFrame],
    difficulty:        Difficulty = "Medium",
    shuffle_autocorr:  bool       = False,
    output_dir:        str        = "output",
    seed:              int        = 42,
) -> dict[str, TickerResult]:
    """Run the anomaly injection pipeline.

    Accepts a dict of {ticker: OHLCV DataFrame} and returns a dict of
    {ticker: TickerResult} with labeled DataFrames and event metadata.

    Args:
        data:             Mapping of ticker → DataFrame with columns
                          [Open, High, Low, Close, Volume] and a DatetimeIndex.
        difficulty:       "Easy", "Medium", or "Hard".
        shuffle_autocorr: If True, shuffle return order within Collective windows.
        output_dir:       Directory to write CSV outputs.
        seed:             Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed=seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: dict[str, TickerResult] = {}

    for ticker, raw_df in data.items():
        logger.info("Processing: %s", ticker)

        df = raw_df.copy().reset_index(drop=False)

        # ── Phase 1 ─────────────────────────────────────────────────────
        sigma = compute_immutable_baseline(df)
        elig_start, elig_end = get_eligible_range(df)
        n_rows = len(df)

        # Target anomaly count (5 % density)
        target_labeled_rows = int(n_rows * TARGET_DENSITY)
        avg_window = {
            "Point": float(POINT_FOOTPRINT),
            "Contextual": (CONTEXTUAL_DURATION[0] + CONTEXTUAL_DURATION[1]) / 2.0,
            "Collective": (COLLECTIVE_DURATION[0] + COLLECTIVE_DURATION[1]) / 2.0,
        }
        total_avg_days = sum(avg_window.values())
        sets = max(1, round(target_labeled_rows / total_avg_days))
        n_each = [sets] * 3

        anomaly_types: list[AnomalyType] = ["Point", "Contextual", "Collective"]
        occupied: set[int] = set()
        events:   list[InjectionEvent] = []

        # ── Phase 2 & 3 — interleave injection attempts ─────────────────
        work_queue: list[AnomalyType] = []
        for atype, count in zip(anomaly_types, n_each):
            work_queue.extend([atype] * count)
        rng.shuffle(work_queue)

        for atype in work_queue:
            param_lo, param_hi = DIFFICULTY_PARAMS[atype][difficulty]

            if atype == "Point":
                window_len = POINT_FOOTPRINT
            elif atype == "Contextual":
                window_len = int(rng.integers(*CONTEXTUAL_DURATION))
            else:
                window_len = int(rng.integers(*COLLECTIVE_DURATION))

            start_idx = sample_injection_index(
                elig_start, elig_end, occupied, window_len, rng
            )
            if start_idx is None:
                logger.warning("  [%s] Skipping %s: no valid slot found.", ticker, atype)
                continue

            end_idx   = start_idx + window_len - 1
            direction = int(rng.choice([-1, 1]))
            magnitude = float(rng.uniform(param_lo, param_hi))

            event = InjectionEvent(
                anomaly_type=atype,
                difficulty=difficulty,
                start_idx=start_idx,
                end_idx=end_idx,
                direction=direction,
                magnitude=magnitude,
            )
            events.append(event)
            mark_occupied(occupied, start_idx, end_idx)

        # ── Apply injections (chronological order) ───────────────────────
        for ev in sorted(events, key=lambda e: e.start_idx):
            if ev.anomaly_type == "Point":
                df = inject_point_anomaly(df, sigma, ev, rng)
            elif ev.anomaly_type == "Contextual":
                df = inject_contextual_anomaly(df, sigma, ev, rng)
            else:
                df = inject_collective_anomaly(df, sigma, ev, rng, shuffle_autocorr)

        # ── Phase 4 ─────────────────────────────────────────────────────
        df = enforce_ohlcv_integrity(df)

        # ── Phase 5 ─────────────────────────────────────────────────────
        df = add_ground_truth_columns(df, events)
        sanity_check(df, ticker)

        # Restore DatetimeIndex if it was preserved during reset_index
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "index" in df.columns:
            df = df.set_index("index")

        # Save
        out_path = Path(output_dir) / f"{ticker}.csv"
        df.to_csv(out_path)

        n_point = sum(e.anomaly_type == "Point" for e in events)
        n_ctx   = sum(e.anomaly_type == "Contextual" for e in events)
        n_coll  = sum(e.anomaly_type == "Collective" for e in events)
        logger.info(
            "  Saved → %s  (%d Point, %d Contextual, %d Collective)",
            out_path, n_point, n_ctx, n_coll,
        )

        results[ticker] = TickerResult(ticker=ticker, df=df, events=events)

    return results


# ---------------------------------------------------------------------------
# Convenience: Synthetic Data Generator (for testing without live data)
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    ticker: str,
    n_days: int = 2765,
    seed:   int = 0,
) -> pd.DataFrame:
    """
    Generates a mean-reverting OHLCV series (Ornstein-Uhlenbeck log-price)
    suitable for testing the pipeline without real market data.
    Produces zero natural anomalies by construction.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n_days)

    # OU log-price
    theta, mu, sigma_ou = 0.03, 5.0, 0.015
    log_p = np.zeros(n_days)
    log_p[0] = mu
    for t in range(1, n_days):
        log_p[t] = log_p[t-1] + theta * (mu - log_p[t-1]) + sigma_ou * rng.standard_normal()

    close  = np.exp(log_p)
    open_  = close * np.exp(rng.normal(0, 0.003, n_days))
    noise  = np.abs(rng.normal(0, 0.005, n_days))
    high   = np.maximum(open_, close) * (1 + noise)
    low    = np.minimum(open_, close) * (1 - noise)
    volume = rng.integers(500_000, 5_000_000, n_days).astype(float)

    return pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume,
    }, index=pd.Index(dates, name="Date"))


def generate_all_synthetic(
    output_dir: str | Path = "../../data/raw_synthetic_data",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for all synthetic tickers in the registry.

    Each ticker gets a unique seed derived from the base seed for reproducibility.
    Also generates a synthetic market benchmark (SPY_SYN) for relative-return features.

    Args:
        output_dir: Directory to save the raw synthetic CSVs.
        seed: Base random seed.

    Returns:
        dict mapping ticker name → DataFrame (with DatetimeIndex).
    """
    tickers = Tickers()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data: dict[str, pd.DataFrame] = {}

    # Generate a synthetic market benchmark
    market_df = generate_synthetic_ohlcv(
        "SPY_SYN", n_days=tickers.n_samples_per_ticker, seed=seed
    )
    market_df.to_csv(out_path / "SPY_SYN.csv")
    logger.info("  Generated synthetic market benchmark: SPY_SYN")

    for i, ticker in enumerate(sorted(tickers.synthetic_tickers)):
        ticker_seed = seed + i + 1
        df = generate_synthetic_ohlcv(
            ticker, n_days=tickers.n_samples_per_ticker, seed=ticker_seed
        )
        df.to_csv(out_path / f"{ticker}.csv")
        data[ticker] = df
        logger.info("  Generated: %s (seed=%d)", ticker, ticker_seed)

    logger.info("Saved %d synthetic tickers → %s", len(data), out_path)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    tickers = Tickers()
    input_path = Path("../../data/raw_tickers_data/")
    tickers_to_inject = tickers.test_tickers

    data = {}
    for i, ticker in enumerate(tickers_to_inject):
        file_path = input_path / f"{ticker}.csv"
        if not file_path.exists():
            logger.warning("File not found: %s", file_path)
            continue
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        data[ticker] = df
        if i == 3:
            break

    if not data:
        logger.error("No data loaded. Exiting.")
    else:
        for diff in DIFFICULTY_LEVELS:
            logger.info("=" * 60)
            logger.info("  Running pipeline — difficulty: %s", diff)
            logger.info("=" * 60)

            run_pipeline(
                data=data,
                difficulty=diff,
                shuffle_autocorr=(diff == "Hard"),
                output_dir=f"../../data/raw_injected_data/{diff}",
                seed=42,
            )

        logger.info("✓ Pipeline complete.")

