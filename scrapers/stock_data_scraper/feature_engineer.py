"""
feature_engineer.py
--------------------
Computes engineered features (log-return, gap-return, Parkinson volatility,
volume Z-score, relative return) on raw OHLCV data and scales them using
RobustScaler. Processes both training and test (injected) data in one
pipeline, fitting separate scalers for each.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from tickers import Tickers

__all__ = ["prepare_df", "run_feature_engineering"]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ROLLING_VOL_WINDOW = 20       # look-back for volume Z-score
TRAIN_SPLIT        = 0.8      # fraction of samples used for scaler fitting
BUFFER_LEN         = 252 + 63 # warm-up buffer used when fitting on test data
VOL_Z_EPS          = 1e-10    # epsilon to avoid division by zero in vol Z


# ── Core ─────────────────────────────────────────────────────────────────────

def prepare_df(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    exclude_price: bool = False,
) -> pd.DataFrame:
    """Compute engineered features on a raw OHLCV DataFrame.

    Features produced:
        f1_log_ret    — log(Close_t / Close_{t-1})
        f2_gap_ret    — log(Open_t  / Close_{t-1})
        f3_parkinson  — Parkinson intra-day volatility estimator
        f4_vol_z      — rolling Z-score of volume
        f5_rel_ret    — log-return minus market log-return (if *market_df* given)

    Args:
        df: Raw OHLCV DataFrame with columns [Date, Open, High, Low, Close, Volume].
        market_df: Optional market-benchmark DataFrame (same schema) for f5_rel_ret.

    Returns:
        DataFrame with raw OHLCV + engineered columns, NaN/inf rows dropped.
    """
    df = df.sort_values("Date").copy()

    # ── Engineered features ──────────────────────────────────────────────
    df["f1_log_ret"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["f2_gap_ret"]   = np.log(df["Open"] / df["Close"].shift(1))
    df["f3_parkinson"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df["High"] / df["Low"])) ** 2
    )

    vol_mean = df["Volume"].rolling(ROLLING_VOL_WINDOW).mean()
    vol_std  = df["Volume"].rolling(ROLLING_VOL_WINDOW).std()
    df["f4_vol_z"] = (df["Volume"] - vol_mean) / (vol_std + VOL_Z_EPS)

    # ── Relative return (market-adjusted) ────────────────────────────────
    if market_df is not None:
        market_df = market_df.sort_values("Date").copy()
        market_df["m_ret"] = np.log(market_df["Close"] / market_df["Close"].shift(1))
        df = df.merge(market_df[["Date", "m_ret"]], on="Date", how="left")
        df["f5_rel_ret"] = df["f1_log_ret"] - df["m_ret"]
        df_out = df.drop(columns=["m_ret"])
    else:
        df_out = df.drop(columns=["f5_rel_ret"], errors="ignore")

    eng_cols = ["f1_log_ret", "f2_gap_ret", "f3_parkinson", "f4_vol_z"]
    raw_cols = ["Open", "High", "Low", "Close", "Volume"] if not exclude_price else []

    if exclude_price:
        # ── Additional features (RSI, MACD, BB Width) ────────────────────────
        delta = df_out["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / (ema_down + 1e-10)
        df_out["f6_rsi_14"] = 100 - (100 / (1 + rs))

        ema_12 = df_out["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_out["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_out["f7_macd_hist"] = macd_line - signal_line

        rolling_mean_20 = df_out["Close"].rolling(20).mean()
        rolling_std_20 = df_out["Close"].rolling(20).std()
        df_out["f8_bb_width"] = (rolling_std_20 * 4) / (rolling_mean_20 + 1e-10)
        
        eng_cols.extend(["f6_rsi_14", "f7_macd_hist", "f8_bb_width"])
    if market_df is not None:
        eng_cols.append("f5_rel_ret")

    result = df_out.dropna().replace([np.inf, -np.inf], 0)
    return result[raw_cols + eng_cols].reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    raw_path: Path | None = None,
    injected_path: Path | None = None,
    train_out_path: Path | None = None,
    test_out_path: Path | None = None,
    analysis_out_path: Path | None = None,
    scaler_path: Path | None = None,
    exclude_price: bool = False,
) -> None:
    """Engineer features & scale for both training and test (injected) data.

    Training data is read from *raw_path*, test data from *injected_path*.
    Each set gets its own fitted RobustScaler, saved separately.

    Args:
        raw_path: Directory with raw {ticker}.csv files (training).
        injected_path: Directory with injected {ticker}_Labeled_{diff}.csv
                       files (test). If None, only training data is processed.
        train_out_path: Output directory for scaled training CSVs.
        test_out_path: Output directory for scaled test CSVs.
        analysis_out_path: Output directory for scaled analysis CSVs.
                           If None, analysis tickers are skipped.
        scaler_path: Directory to persist fitted scaler dicts.
    """
    tickers = Tickers()
    raw_dir          = raw_path or Path("../../data/raw_tickers_data")
    injected_dir     = injected_path or Path("../../data/raw_injected_data")
    train_out_dir    = train_out_path
    test_out_dir     = test_out_path or Path("../../data/engineered_test_data")
    analysis_out_dir = analysis_out_path
    scaler_dir    = scaler_path or Path("../../utils")

    train_end = int(tickers.n_samples_per_ticker * TRAIN_SPLIT)

    # Look for market benchmark (SPY or SPY_SYN for synthetic data)
    market_csv = raw_dir / f"{tickers.market_ticker}.csv"
    if not market_csv.exists():
        market_csv = raw_dir / "SPY_SYN.csv"
    if not market_csv.exists():
        logger.error("Market benchmark not found in %s. Run scraper first.", raw_dir)
        return
    market_df = pd.read_csv(market_csv)

    dirs_to_create = [test_out_dir, scaler_dir]
    if train_out_dir:
        dirs_to_create.append(train_out_dir)
    if analysis_out_dir:
        dirs_to_create.append(analysis_out_dir)
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # ── Process training tickers ─────────────────────────────────────────
    if train_out_dir is not None:
        logger.info("Engineering features — training set")
        train_scalers: dict[str, RobustScaler] = {}

        for ticker in tqdm(sorted(tickers.training_tickers), desc="Train"):
            csv_path = raw_dir / f"{ticker}.csv"
            if not csv_path.exists():
                logger.warning("  ✗ %s: CSV not found, skipping", ticker)
                continue

            df = pd.read_csv(csv_path)
            df_eng = prepare_df(df, market_df, exclude_price=exclude_price)
            df_eng = df_eng.iloc[-tickers.n_samples_per_ticker:].reset_index(drop=True)

            scaler = RobustScaler()
            scaler.fit(df_eng.iloc[:train_end])

            scaled = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)
            scaled.to_csv(train_out_dir / f"{ticker}.csv", index=False)
            train_scalers[ticker] = scaler

        joblib.dump(train_scalers, scaler_dir / "train_scalers.joblib", compress=3)
        logger.info("Saved %d train scalers → %s", len(train_scalers), scaler_dir / "train_scalers.joblib")
    else:
        logger.info("Skipping training set (train_out_path=None)")

    # ── Process analysis tickers ─────────────────────────────────────────
    if analysis_out_dir is not None:
        logger.info("Engineering features — analysis set")
        analysis_scalers: dict[str, RobustScaler] = {}

        for ticker in tqdm(sorted(tickers.analysis_tickers), desc="Analysis"):
            csv_path = raw_dir / f"{ticker}.csv"
            if not csv_path.exists():
                logger.warning("  ✗ %s: CSV not found, skipping", ticker)
                continue

            df = pd.read_csv(csv_path)
            df_eng = prepare_df(df, market_df, exclude_price=exclude_price)
            df_eng = df_eng.iloc[-tickers.n_samples_per_ticker:].reset_index(drop=True)

            scaler = RobustScaler()
            scaler.fit(df_eng.iloc[:train_end])

            scaled = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)
            scaled.to_csv(analysis_out_dir / f"{ticker}.csv", index=False)
            analysis_scalers[ticker] = scaler

        joblib.dump(analysis_scalers, scaler_dir / "analysis_scalers.joblib", compress=3)
        logger.info("Saved %d analysis scalers → %s", len(analysis_scalers), scaler_dir / "analysis_scalers.joblib")
    else:
        logger.info("Skipping analysis set (analysis_out_path=None)")

    # ── Process test (injected) tickers ──────────────────────────────────
    if not injected_dir.exists():
        logger.info("No injected data dir (%s) — skipping test set.", injected_dir)
        return

    injected_files = sorted(injected_dir.glob("**/*.csv"))
    if not injected_files:
        logger.info("No injected CSVs found in %s — skipping test set.", injected_dir)
        return

    logger.info("Engineering features — test set (%d files)", len(injected_files))
    test_scalers: dict[str, RobustScaler] = {}

    for csv_path in tqdm(injected_files, desc="Test"):
        df = pd.read_csv(csv_path)
        # Preserve label columns if present
        label_cols = [c for c in df.columns if c.startswith("Is_Anomaly") or c == "anomaly"]
        labels = df[label_cols] if label_cols else None

        df_eng = prepare_df(df, market_df, exclude_price=exclude_price)
        df_eng = df_eng.iloc[-tickers.n_samples_per_ticker:].reset_index(drop=True)

        scaler = RobustScaler()
        scaler.fit(df_eng.iloc[:BUFFER_LEN])

        scaled = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)

        # Re-attach labels
        if labels is not None:
            labels = labels.iloc[-len(scaled):].reset_index(drop=True)
            scaled = pd.concat([scaled, labels], axis=1)

        # Mirror subdirectory structure (e.g. Easy/, Medium/, Hard/) in output
        rel = csv_path.relative_to(injected_dir)
        out_file = test_out_dir / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        scaled.to_csv(out_file, index=False)
        test_scalers[csv_path.stem] = scaler

    joblib.dump(test_scalers, scaler_dir / "test_scalers.joblib", compress=3)
    logger.info("Saved %d test scalers → %s", len(test_scalers), scaler_dir / "test_scalers.joblib")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_feature_engineering()