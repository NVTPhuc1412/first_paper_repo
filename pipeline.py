"""
pipeline.py
-----------
Centralized anomaly analysis pipeline.

Usage:
    python pipeline.py NVDA
    python pipeline.py TSLA --skip-llm
    python pipeline.py INTC --dry-run

Pipeline stages:
    1. Fetch stock data (yfinance)
    2. Feature engineering (log-return, gap-return, etc.)
    3. Anomaly score extraction (AT + TranAD)
    4. XAI analysis (Integrated Gradients + TimeSHAP)
    5. LLM Attribution (Gemini + GDELT headlines)
"""

import os
import sys
import argparse
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline_config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(cfg: PipelineConfig) -> None:
    """Execute the full anomaly analysis pipeline for a single ticker.

    Args:
        cfg: Pipeline configuration (ticker, paths, flags).
    """
    ticker = cfg.ticker_upper
    print(f"\n{'='*70}")
    print(f"  ANOMALY ANALYSIS PIPELINE — {ticker}")
    print(f"{'='*70}\n")

    cfg.ensure_dirs()

    # ── Stage 1: Fetch Stock Data ─────────────────────────────────────────
    print(f"[1/5] Fetching stock data for {ticker}...")

    from scrapers_n_preprocessor.fetch import fetch_stock_data

    try:
        ticker_df, market_df = fetch_stock_data(
            ticker=ticker,
            market_ticker=cfg.market_ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            min_rows=cfg.n_samples,
            save_dir=cfg.raw_out,
        )
        print(f"  ✓ {ticker}: {len(ticker_df)} rows | {cfg.market_ticker}: {len(market_df)} rows\n")
    except ValueError as e:
        print(f"  ✗ {e}")
        return

    # ── Stage 2: Feature Engineering ──────────────────────────────────────
    print(f"[2/5] Engineering features...")

    from scrapers_n_preprocessor.feature_engineer_single import engineer_features

    scaled_df, scaler, feature_names = engineer_features(
        ticker_df=ticker_df,
        market_df=market_df,
        n_samples=cfg.n_samples,
        train_split=cfg.train_split,
        save_dir=cfg.features_out,
        scaler_dir=cfg.features_out,
        ticker_name=ticker,
    )
    print(f"  ✓ {len(scaled_df)} rows, {len(feature_names)} features\n")

    if cfg.dry_run:
        print("[DRY RUN] Stopping after feature engineering.")
        print(f"  Scaled CSV: {cfg.features_out / f'{ticker}.csv'}")
        print(f"  Raw data:   {cfg.raw_out / f'{ticker}.csv'}")
        return

    # ── Stage 3: Anomaly Score Extraction ─────────────────────────────────
    print(f"[3/5] Extracting anomaly scores...")

    from extract_anomaly_scores import run_scoring_pipeline

    # Prepare model configs with absolute paths
    model_configs = []
    for m in cfg.models:
        mc = dict(m)
        mc['path'] = str(cfg.model_path(m))
        mc['threshold_path'] = str(cfg.threshold_path(m)) if cfg.threshold_path(m).exists() else None
        model_configs.append(mc)

    scores_df = run_scoring_pipeline(
        ticker=ticker,
        data_dir=str(cfg.features_out),
        raw_data_dir=str(cfg.raw_out),
        scaler_path=str(cfg.features_out),
        out_dir=str(cfg.scores_out),
        models=model_configs,
    )
    print(f"  ✓ Scores extracted\n")

    # ── Stage 3.5: Generate Plots ─────────────────────────────────────────
    print(f"  Generating plots...")

    import pandas as pd
    from plot_pipeline import plot_anomaly_scores, plot_price_reconstruction, plot_combined

    raw_df = pd.read_csv(cfg.raw_out / f"{ticker}.csv")

    plot_anomaly_scores(scores_df, ticker, model_configs, str(cfg.plots_out))
    plot_price_reconstruction(scores_df, raw_df, ticker, model_configs, str(cfg.plots_out))
    plot_combined(scores_df, raw_df, ticker, model_configs, str(cfg.plots_out))
    print()

    # ── Stage 4: XAI Analysis ────────────────────────────────────────────
    print(f"[4/5] Running XAI analysis (IG + TimeSHAP)...")

    from xai_analysis import run_xai_for_ticker

    event_results = run_xai_for_ticker(
        ticker=ticker,
        scores_df=scores_df,
        data_dir=str(cfg.features_out),
        raw_data_dir=str(cfg.raw_out),
        out_dir=str(cfg.xai_out),
        cache_dir=str(cfg.xai_cache),
        models=model_configs,
        top_n=cfg.top_n_events,
        ig_n_steps=cfg.ig_n_steps,
        timeshap_n_samples=cfg.timeshap_n_samples,
    )
    print(f"  ✓ XAI analysis complete\n")

    # ── Stage 5: LLM Attribution ──────────────────────────────────────────
    if cfg.skip_llm:
        print(f"[5/5] LLM Attribution — SKIPPED (--skip-llm)\n")
    else:
        print(f"[5/5] LLM Attribution...")

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print(f"  ✗ GEMINI_API_KEY not set — skipping attribution.")
            print(f"    Set it with: set GEMINI_API_KEY=your_key_here\n")
        else:
            from xai_analysis.news_fetcher import generate_gdelt_query, fetch_all_event_headlines
            from xai_analysis.attribute import run_attribution_pipeline
            from google import genai

            client = genai.Client(api_key=api_key)

            # Generate GDELT query
            print(f"  Generating GDELT search query for {ticker}...")
            gdelt_query = generate_gdelt_query(ticker, client, model=cfg.llm_model)
            print(f"  Query: {gdelt_query[:80]}...")

            # Collect all event dates
            all_event_dates = set()
            for label, events in event_results.items():
                for ev in events:
                    all_event_dates.add(ev['date'])

            # Fetch headlines
            print(f"  Fetching headlines for {len(all_event_dates)} events...")
            headlines_by_date = fetch_all_event_headlines(
                query=gdelt_query,
                event_dates=sorted(all_event_dates),
                lookback_hours=cfg.lookback_hours,
                lookahead_hours=cfg.lookahead_hours,
                cache_dir=str(cfg.news_cache),
                ticker=ticker,
            )

            total_headlines = sum(len(h) for h in headlines_by_date.values())
            print(f"  ✓ {total_headlines} headlines across {len(headlines_by_date)} events")

            # Run attribution
            print(f"  Running LLM attribution...")
            run_attribution_pipeline(
                ticker=ticker,
                event_results=event_results,
                headlines_by_date=headlines_by_date,
                out_dir=str(cfg.attribution_out),
                models=model_configs,
                llm_model=cfg.llm_model,
            )
            print(f"  ✓ Attribution complete\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  PIPELINE COMPLETE — {ticker}")
    print(f"{'='*70}")
    print(f"  Output directory: {cfg.pipeline_out}")
    print(f"    ├── raw/           Raw OHLCV data")
    print(f"    ├── features/      Scaled feature CSVs + scalers")
    print(f"    ├── scores/        Anomaly scores CSV")
    print(f"    ├── plots/         Score + price plots")
    print(f"    ├── xai/           IG heatmaps + TimeSHAP plots")
    print(f"    └── attribution/   LLM attribution reports")
    print(f"  Cache:")
    print(f"    ├── data/xai_cache/{ticker}/    XAI results (.npz)")
    print(f"    └── data/news_cache/{ticker}/   GDELT headlines (.csv)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Centralized Anomaly Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline.py NVDA                 # Full pipeline
    python pipeline.py TSLA --skip-llm      # Skip LLM attribution
    python pipeline.py INTC --dry-run       # Only fetch + engineer features
        """,
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. NVDA, TSLA)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip the LLM attribution stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run fetch + feature engineering (no GPU)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top anomaly events to analyze (default: 5)")

    args = parser.parse_args()

    cfg = PipelineConfig(
        ticker=args.ticker,
        skip_llm=args.skip_llm,
        dry_run=args.dry_run,
        top_n_events=args.top_n,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
