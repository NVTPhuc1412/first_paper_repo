"""
pipeline_config.py
------------------
Centralized configuration for the anomaly analysis pipeline.
All paths, model definitions, and default settings live here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Project root (directory containing this file) ────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    """All settings for a single pipeline run."""

    # ── Ticker ────────────────────────────────────────────────────────────
    ticker: str = ""
    market_ticker: str = "SPY"

    # ── Data parameters ───────────────────────────────────────────────────
    n_samples: int = 2765              # ~10 years of trading days
    start_date: str = "2014-01-01"     # extra warm-up year
    end_date: str = "2025-12-31"
    train_split: float = 0.8

    # ── Directories ───────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    best_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "best_saved_model")

    # ── Model definitions ─────────────────────────────────────────────────
    models: list = field(default_factory=lambda: [
        {
            "label":    "AT",
            "name":     "Anomaly Transformer",
            "encoder":  None,
            "detector": "Anomaly Transformer",
            "weights":  "None_Anomaly Transformer.pth",
            "threshold": "None_Anomaly Transformer_percentile_thresholds.npy",
            "score_col": "anomaly_score_AT",
            "pred_col":  "prediction_AT",
        },
        {
            "label":    "TranAD",
            "name":     "TimesNet + TranAD",
            "encoder":  "TimesNet",
            "detector": "TranAD",
            "weights":  "TimesNet_TranAD.pth",
            "threshold": "TimesNet_TranAD_pot_thresholds.npy",
            "score_col": "anomaly_score_TranAD",
            "pred_col":  "prediction_TranAD",
        },
    ])

    # ── XAI settings ─────────────────────────────────────────────────────
    top_n_events: int = 5
    ig_n_steps: int = 100
    timeshap_n_samples: int = 50

    # ── News / LLM Attribution ────────────────────────────────────────────
    lookback_hours: int = 72
    lookahead_hours: int = 24
    llm_model: str = "gemini-3.1-flash-lite-preview"
    gdelt_between_requests: int = 6    # seconds between GDELT API calls

    # ── Flags ─────────────────────────────────────────────────────────────
    skip_llm: bool = False
    dry_run: bool = False              # only fetch + feature engineer

    # ── Derived paths (computed after init) ────────────────────────────────
    def __post_init__(self):
        self.ticker_upper = self.ticker.upper()

        # Pipeline output dirs
        self.pipeline_out = self.results_dir / "pipeline" / self.ticker_upper
        self.raw_out = self.pipeline_out / "raw"
        self.features_out = self.pipeline_out / "features"
        self.scores_out = self.pipeline_out / "scores"
        self.xai_out = self.pipeline_out / "xai"
        self.attribution_out = self.pipeline_out / "attribution"
        self.plots_out = self.pipeline_out / "plots"

        # Cache dirs
        self.xai_cache = self.data_dir / "xai_cache" / self.ticker_upper
        self.news_cache = self.data_dir / "news_cache" / self.ticker_upper

    def model_path(self, model_cfg: dict) -> Path:
        return self.best_model_dir / model_cfg["weights"]

    def threshold_path(self, model_cfg: dict) -> Path:
        return self.best_model_dir / model_cfg["threshold"]

    def ensure_dirs(self):
        """Create all output and cache directories."""
        for d in [
            self.pipeline_out, self.raw_out, self.features_out,
            self.scores_out, self.xai_out, self.attribution_out,
            self.plots_out, self.xai_cache, self.news_cache,
        ]:
            d.mkdir(parents=True, exist_ok=True)
