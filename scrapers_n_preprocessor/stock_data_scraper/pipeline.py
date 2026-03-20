"""
pipeline.py
------------
Centralized orchestrator for the stock data pipeline.

Stages (in order / standalone):
  1. scrape   — download raw OHLCV from Yahoo Finance
  2. check    — filter tickers by stationarity + rank stability
  3. inject   — inject synthetic anomalies into test tickers
  4. engineer — compute features & scale (training + test data)
  plot        — generate comparison charts from saved data
  all         — run stages 1-4 sequentially

Usage:
  python pipeline.py scrape
  python pipeline.py check
  python pipeline.py inject --difficulty Medium
  python pipeline.py engineer
  python pipeline.py plot --difficulty all
  python pipeline.py all
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib
import pandas as pd

from tickers import Tickers
from data_scraper import run_scraper
from stationary_check import run_stationarity_check
from anomaly_injector import (
    run_pipeline as run_injection,
    generate_all_synthetic,
    DIFFICULTY_LEVELS,
)
from feature_engineer import run_feature_engineering

logger = logging.getLogger(__name__)

# ── Default Paths ────────────────────────────────────────────────────────────

RAW_DATA_DIR            = Path("../../data/raw_tickers_data")
INJECTED_DATA_DIR       = Path("../../data/raw_injected_data")
SYNTHETIC_DATA_DIR      = Path("../../data/raw_synthetic_data")
INJECTED_SYNTHETIC_DIR  = Path("../../data/raw_injected_synthetic")
TRAIN_ENGINEERED        = Path("../../data/engineered_train_data")
TEST_ENGINEERED         = Path("../../data/engineered_test_data")
ANALYSIS_ENGINEERED     = Path("../../data/engineered_analysis_data")
TEST_ENGINEERED_SYN     = Path("../../data/engineered_test_synthetic")
SCALER_DIR              = Path("../../utils")
PLOT_DIR                = Path("output")


# ── Stage Runners ────────────────────────────────────────────────────────────

def stage_scrape() -> None:
    """Stage 1: Download raw OHLCV data."""
    logger.info("=" * 60)
    logger.info("  STAGE 1 — Scraping raw OHLCV data")
    logger.info("=" * 60)
    run_scraper(output_dir=RAW_DATA_DIR)


def stage_check() -> None:
    """Stage 2: Stationarity filter + stability ranking."""
    logger.info("=" * 60)
    logger.info("  STAGE 2 — Stationarity check & stability ranking")
    logger.info("=" * 60)
    run_stationarity_check(raw_data_dir=RAW_DATA_DIR)


def stage_inject(difficulty: str = "Medium", synthetic: bool = False) -> None:
    """Stage 3: Anomaly injection into test tickers.

    Args:
        difficulty: Run for a single difficulty, or "all" for Easy/Medium/Hard.
        synthetic: If True, generate synthetic OU data and inject into that.
    """
    from anomaly_injector import Tickers as _T

    tickers = _T()

    if synthetic:
        logger.info("Generating synthetic OHLCV data...")
        data = generate_all_synthetic(output_dir=SYNTHETIC_DATA_DIR)
        out_dir = INJECTED_SYNTHETIC_DIR
    else:
        data: dict[str, pd.DataFrame] = {}
        for ticker in sorted(tickers.test_tickers):
            csv_path = RAW_DATA_DIR / f"{ticker}.csv"
            if not csv_path.exists():
                logger.warning("  ✗ %s: CSV not found, skipping", ticker)
                continue
            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            data[ticker] = df
        out_dir = INJECTED_DATA_DIR

    if not data:
        logger.error("No data available. Run 'scrape' first or use --synthetic.")
        return

    difficulties = DIFFICULTY_LEVELS if difficulty == "all" else [difficulty]

    for diff in difficulties:
        label = f"{diff} (SYNTHETIC)" if synthetic else diff
        logger.info("=" * 60)
        logger.info("  Running injection — difficulty: %s", label)
        logger.info("=" * 60)

        run_injection(
            data=data,
            difficulty=diff,
            shuffle_autocorr=(diff == "Hard"),
            output_dir=str(out_dir / diff),
            seed=42,
        )


def _events_from_labels(df: pd.DataFrame, difficulty: str) -> list:
    """Reconstruct InjectionEvent spans from the label columns in a saved CSV."""
    from anomaly_injector import InjectionEvent

    events: list[InjectionEvent] = []
    label_map = {
        "Is_Anomaly_Point":      "Point",
        "Is_Anomaly_Contextual": "Contextual",
        "Is_Anomaly_Collective": "Collective",
    }
    for col, atype in label_map.items():
        if col not in df.columns:
            continue
        mask = df[col].astype(bool)
        # Find contiguous True runs
        groups = (mask != mask.shift()).cumsum()
        for _, grp in df[mask].groupby(groups[mask]):
            events.append(InjectionEvent(
                anomaly_type=atype,
                difficulty=difficulty,
                start_idx=grp.index[0],
                end_idx=grp.index[-1],
                direction=0,   # unknown from saved labels
                magnitude=0.0, # unknown from saved labels
            ))
    return events


def stage_plot(difficulty: str = "all") -> None:
    """Generate comparison charts from previously saved raw + injected CSVs.

    Reads raw data from RAW_DATA_DIR and labeled data from
    INJECTED_DATA_DIR/{difficulty}/, reconstructing anomaly event
    spans from the label columns so that shading works correctly.

    Args:
        difficulty: "Easy", "Medium", "Hard", or "all".
    """
    matplotlib.use("Agg")
    from plots import plot_comparison

    difficulties = DIFFICULTY_LEVELS if difficulty == "all" else [difficulty]

    for diff in difficulties:
        diff_dir = INJECTED_DATA_DIR / diff
        if not diff_dir.exists():
            logger.warning("No injected directory for %s at %s — skipping.", diff, diff_dir)
            continue

        injected_files = sorted(diff_dir.glob("*.csv"))
        if not injected_files:
            logger.warning("No injected CSVs in %s — skipping.", diff_dir)
            continue

        logger.info("=" * 60)
        logger.info("  Plotting — difficulty: %s  (%d files)", diff, len(injected_files))
        logger.info("=" * 60)

        for csv_path in injected_files:
            ticker = csv_path.stem

            raw_csv = RAW_DATA_DIR / f"{ticker}.csv"
            if not raw_csv.exists():
                logger.warning("  ✗ %s: raw CSV not found at %s — skipping.", ticker, raw_csv)
                continue

            original = pd.read_csv(raw_csv, parse_dates=["Date"], index_col="Date")
            labeled  = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            events   = _events_from_labels(labeled.reset_index(), diff)

            plot_comparison(
                original=original,
                labeled=labeled,
                events=events,
                ticker=ticker,
                difficulty=diff,
                output_dir=str(PLOT_DIR / diff),
            )
            logger.info("  ✓ %s plotted.", ticker)


def stage_engineer(exclude_price: bool = False, synthetic: bool = False) -> None:
    """Stage 4: Feature engineering + scaling (train + test).

    When synthetic=True, only the test set is engineered (from injected
    synthetic data). Training always uses real tickers.
    """
    logger.info("=" * 60)
    logger.info(f"  STAGE 4 — Feature engineering & scaling (exclude_price={exclude_price}, synthetic={synthetic})")
    logger.info("=" * 60)

    if synthetic:
        # Synthetic mode: engineer test data only (model trains on real data)
        test_out = TEST_ENGINEERED_SYN
        if exclude_price:
            test_out = test_out.with_name(test_out.name + "_stat")
        scaler_out = SCALER_DIR / "scalers_syn"

        run_feature_engineering(
            raw_path=SYNTHETIC_DATA_DIR,
            injected_path=INJECTED_SYNTHETIC_DIR,
            train_out_path=None,
            test_out_path=test_out,
            analysis_out_path=None,
            scaler_path=scaler_out,
            exclude_price=exclude_price,
        )
    else:
        # Standard mode: engineer both train and test from real data
        train_out = TRAIN_ENGINEERED
        test_out = TEST_ENGINEERED
        analysis_out = ANALYSIS_ENGINEERED
        scaler_out = SCALER_DIR

        if exclude_price:
            train_out = train_out.with_name(train_out.name + "_stat")
            test_out = test_out.with_name(test_out.name + "_stat")
            analysis_out = analysis_out.with_name(analysis_out.name + "_stat")
            scaler_out = SCALER_DIR / "scalers_stat"

        run_feature_engineering(
            raw_path=RAW_DATA_DIR,
            injected_path=INJECTED_DATA_DIR,
            train_out_path=train_out,
            test_out_path=test_out,
            analysis_out_path=analysis_out,
            scaler_path=scaler_out,
            exclude_price=exclude_price,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Stock data pipeline: scrape → check → inject → engineer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    sub.add_parser("scrape", help="Download raw OHLCV data from Yahoo Finance")

    # check
    sub.add_parser("check", help="Stationarity filter + stability ranking")

    # inject
    p_inject = sub.add_parser("inject", help="Inject synthetic anomalies")
    p_inject.add_argument(
        "--difficulty", default="all",
        choices=DIFFICULTY_LEVELS + ["all"],
        help="Difficulty level (default: all)",
    )
    p_inject.add_argument("--synthetic", action="store_true", help="Use synthetic OU data")

    # engineer
    p_eng = sub.add_parser("engineer", help="Feature engineering & scaling (train + test)")
    p_eng.add_argument("--exclude-price", action="store_true", help="Exclude raw OHLC prices")
    p_eng.add_argument("--synthetic", action="store_true", help="Use synthetic data paths")

    # plot (standalone — reads saved CSVs)
    p_plot = sub.add_parser("plot", help="Generate comparison charts from saved data")
    p_plot.add_argument(
        "--difficulty", default="all",
        choices=DIFFICULTY_LEVELS + ["all"],
        help="Difficulty level to plot (default: all)",
    )

    # all
    p_all = sub.add_parser("all", help="Run all stages sequentially")
    p_all.add_argument("--exclude-price", action="store_true", help="Exclude raw OHLC prices")
    p_all.add_argument("--synthetic", action="store_true", help="Use synthetic OU data")

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    args = build_parser().parse_args()
    synthetic = getattr(args, "synthetic", False)
    exclude_price = getattr(args, "exclude_price", False)

    if args.command == "scrape":
        stage_scrape()

    elif args.command == "check":
        stage_check()

    elif args.command == "inject":
        stage_inject(difficulty=args.difficulty, synthetic=synthetic)

    elif args.command == "engineer":
        stage_engineer(exclude_price=exclude_price, synthetic=synthetic)

    elif args.command == "plot":
        stage_plot(difficulty=args.difficulty)

    elif args.command == "all":
        if not synthetic:
            stage_scrape()
            stage_check()
        stage_inject(difficulty="all", synthetic=synthetic)
        stage_engineer(exclude_price=exclude_price, synthetic=synthetic)

    logger.info("✓ Done.")


if __name__ == "__main__":
    main()
