"""
plots.py
--------
Light-mode comparison charts for the anomaly injection pipeline.
Optional debug utility — not part of the core data pipeline.
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from anomaly_injector import InjectionEvent

__all__ = ["plot_comparison"]

logger = logging.getLogger(__name__)

# ── Colour palette (high-contrast on white) ──────────────────────────────────

ANOMALY_COLORS = {
    "Point":       ("#dc2626", "#fecaca"),   # red  (dark, light)
    "Contextual":  ("#d97706", "#fde68a"),   # amber
    "Collective":  ("#4f46e5", "#c7d2fe"),   # indigo
}


def plot_comparison(
    original,
    labeled,
    events:      list[InjectionEvent],
    ticker:      str,
    difficulty:  str = "Medium",
    output_dir:  str = "output",
) -> None:
    """Produce a 3-panel comparison figure for a single ticker (light mode).

    Panels:
      1. Close price — original vs transformed, anomaly windows shaded
      2. High−Low range — original vs transformed (volatility changes)
      3. Volume — original vs transformed

    Saves to ``{output_dir}/{ticker}_comparison_{difficulty}.png``.
    """
    idx = original.index

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.2]},
    )

    # ── Light-mode styling ───────────────────────────────────────────────
    fig.patch.set_facecolor("#ffffff")
    for ax in axes:
        ax.set_facecolor("#fafafa")
        ax.tick_params(colors="#374151", labelsize=8)
        ax.yaxis.label.set_color("#374151")
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")
        ax.grid(True, color="#e5e7eb", linewidth=0.5, alpha=0.8)

    # ── Shared anomaly shading ───────────────────────────────────────────
    def shade_events(ax):
        for ev in events:
            s = ev.start_idx
            e = min(ev.end_idx, len(idx) - 1)
            dark, light = ANOMALY_COLORS[ev.anomaly_type]
            ax.axvspan(idx[s], idx[e], alpha=0.15, color=light, zorder=1)
            ax.axvline(idx[s], color=dark, linewidth=0.6, alpha=0.6, zorder=2)

    # ── Panel 1: Close price ─────────────────────────────────────────────
    ax = axes[0]
    shade_events(ax)
    ax.plot(idx, original["Close"], color="#9ca3af", linewidth=1.0,
            label="Original", zorder=3, alpha=0.85)
    ax.plot(idx, labeled["Close"],  color="#2563eb", linewidth=1.1,
            label="Transformed", zorder=4)
    ax.set_ylabel("Close Price", fontsize=9)
    ax.set_title(
        f"{ticker}  ·  Anomaly Injection Comparison  ·  {difficulty}",
        color="#111827", fontsize=12, fontweight="bold", pad=10,
    )

    # Legend for anomaly types
    patch_handles = [
        mpatches.Patch(color=ANOMALY_COLORS[t][0], alpha=0.9, label=t)
        for t in ["Point", "Contextual", "Collective"]
        if any(ev.anomaly_type == t for ev in events)
    ]
    line_handles = [
        plt.Line2D([0], [0], color="#9ca3af", linewidth=1.4, label="Original"),
        plt.Line2D([0], [0], color="#2563eb", linewidth=1.4, label="Transformed"),
    ]
    ax.legend(
        handles=line_handles + patch_handles,
        loc="upper left", fontsize=8,
        facecolor="#ffffff", edgecolor="#d1d5db", labelcolor="#374151",
    )

    # ── Panel 2: High-Low range ──────────────────────────────────────────
    ax = axes[1]
    shade_events(ax)
    orig_range = original["High"] - original["Low"]
    new_range  = labeled["High"]  - labeled["Low"]
    ax.fill_between(idx, orig_range, alpha=0.35, color="#9ca3af", label="Original HL range")
    ax.fill_between(idx, new_range,  alpha=0.50, color="#f59e0b", label="Transformed HL range")
    ax.set_ylabel("High − Low", fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5,
              facecolor="#ffffff", edgecolor="#d1d5db", labelcolor="#374151")

    # ── Panel 3: Volume ──────────────────────────────────────────────────
    ax = axes[2]
    shade_events(ax)
    ax.bar(idx, original["Volume"] / 1e6, width=1, color="#d1d5db",
           alpha=0.6, label="Original volume")
    ax.bar(idx, labeled["Volume"]  / 1e6, width=1, color="#6366f1",
           alpha=0.7, label="Transformed volume")
    ax.set_ylabel("Volume (M)", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax.legend(loc="upper left", fontsize=7.5,
              facecolor="#ffffff", edgecolor="#d1d5db", labelcolor="#374151")

    # ── x-axis date formatting ───────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 1])
    out_path = Path(output_dir) / f"{ticker}_comparison_{difficulty}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("  Chart → %s", out_path)
