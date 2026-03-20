"""
xai_analysis.py
================
Explainable AI analysis for the anomaly detection pipeline.

Implements two XAI techniques:
  1. Integrated Gradients (via Captum) — feature-level attribution
  2. TimeSHAP (manual implementation) — temporal importance of lookback steps

Targets the top anomaly events (by score) for each of the two selected models:
  - None + Anomaly Transformer  (best overall)
  - TimesNet + TranAD            (best point anomaly precision)

Outputs:
  - Per-event heatmap images  → ./results/xai/
  - Summary markdown report   → ./results/xai_report.md

Requirements:
    pip install captum matplotlib seaborn
"""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from captum.attr import IntegratedGradients

from model.config import Config
from model.main_model import Model
from model.utils import my_kl_loss

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR         = './data/engineered_analysis_data_stat'
RAW_DATA_DIR     = './data/raw_tickers_data'
SCORES_DIR       = './data/extracted_scores'
OUT_DIR          = './results/xai'
REPORT_PATH      = './results/xai_report.md'

TICKERS          = ['NVDA', 'TSLA', 'INTC']
TOP_N_EVENTS     = 5          # top anomaly events per model per ticker

# Model definitions: (label, encoder, detector, weights_path)
MODELS = [
    {
        'label':    'AT',
        'name':     'Anomaly Transformer',
        'encoder':  None,
        'detector': 'Anomaly Transformer',
        'path':     './best_saved_model/None_Anomaly Transformer.pth',
    },
    {
        'label':    'TranAD',
        'name':     'TimesNet + TranAD',
        'encoder':  'TimesNet',
        'detector': 'TranAD',
        'path':     './best_saved_model/TimesNet_TranAD.pth',
    },
]


# ── Differentiable Score Wrappers ─────────────────────────────────────────────
# Captum's IntegratedGradients requires a single differentiable scalar output.
# These wrappers replicate the scoring logic from scorer.py but keep gradients.

class ATScoreWrapper(nn.Module):
    """Wraps the full Model to produce a differentiable scalar anomaly score
    for the Anomaly Transformer (averaged across the window)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = 50

    def forward(self, x):
        """
        Args:
            x: [B, L, C]
        Returns:
            score: [B] — mean anomaly score across the window
        """
        enc_out, series_list, prior_list, _ = self.model(x)

        # Reconstruction loss: mean MSE per timestep
        rec_loss = torch.mean((x - enc_out) ** 2, dim=-1)  # [B, L]

        # Association discrepancy
        win_size = x.shape[1]
        series_loss = None
        prior_loss = None

        for u in range(len(prior_list)):
            prior_norm = prior_list[u] / torch.unsqueeze(
                torch.sum(prior_list[u], dim=-1), dim=-1
            ).repeat(1, 1, 1, win_size)

            kl_sp = my_kl_loss(series_list[u], prior_norm.detach()) * self.temperature
            kl_ps = my_kl_loss(prior_norm, series_list[u].detach()) * self.temperature

            if series_loss is None:
                series_loss = kl_sp
                prior_loss = kl_ps
            else:
                series_loss = series_loss + kl_sp
                prior_loss = prior_loss + kl_ps

        assoc = series_loss + prior_loss            # [B, L]
        metric = torch.softmax(-assoc, dim=-1)      # [B, L]
        score = metric * rec_loss                   # [B, L]

        return score.mean(dim=-1)                   # [B]


class TranADScoreWrapper(nn.Module):
    """Wraps the full Model to produce a differentiable scalar anomaly score
    for TranAD (mean MSE of the last timestep across features)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Args:
            x: [B, L, C]
        Returns:
            score: [B] — mean reconstruction error at the last timestep
        """
        x1, x2 = self.model(x)
        tgt = x[:, -1:, :]                 # [B, 1, C]
        mse = 0.5 * (x1 - tgt)**2 + 0.5 * (x2 - tgt)**2  # [B, 1, C]
        return mse.mean(dim=(1, 2))         # [B]


# ── Data Loading Helpers ──────────────────────────────────────────────────────

def load_model(model_cfg, enc_in, device):
    """Instantiate and load a trained model."""
    config = Config(
        encoder=model_cfg['encoder'],
        detector=model_cfg['detector'],
        enc_in=enc_in,
    )
    config.device = device
    model = Model(config).to(device)
    model.load_state_dict(
        torch.load(model_cfg['path'], map_location=device, weights_only=True)
    )
    model.eval()
    return model, config


def load_ticker_data(ticker):
    """Load the engineered feature CSV for a ticker and return tensor + dates."""
    csv_path = os.path.join(DATA_DIR, f'{ticker}.csv')
    df = pd.read_csv(csv_path)
    value_cols = [
        c for c in df.columns
        if c not in ['Date', 'ticker', 'anomaly'] and not c.startswith('Is_Anomaly')
    ]

    # Load dates from raw data
    raw_csv = os.path.join(RAW_DATA_DIR, f'{ticker}.csv')
    raw_df = pd.read_csv(raw_csv, usecols=['Date'])
    dates = raw_df['Date'].iloc[-len(df):].reset_index(drop=True)

    data = torch.from_numpy(df[value_cols].to_numpy().astype(np.float32))
    return data, dates, value_cols


def load_scores(ticker):
    """Load pre-computed anomaly scores for a ticker."""
    path = os.path.join(SCORES_DIR, f'{ticker}_scores.csv')
    return pd.read_csv(path, parse_dates=['Date'])


def get_top_events(score_df, score_col, n):
    """Return the indices and dates of the top-N anomaly scores."""
    valid = score_df.dropna(subset=[score_col])
    top = valid.nlargest(n, score_col)
    return top[['Date', score_col]].reset_index()


# ── Integrated Gradients ──────────────────────────────────────────────────────

def run_integrated_gradients(wrapper, window, feature_names, n_steps=100):
    """
    Run Integrated Gradients on a single input window.

    Args:
        wrapper:       ATScoreWrapper or TranADScoreWrapper
        window:        [1, L, C] tensor (requires_grad will be set)
        feature_names: list of str
        n_steps:       IG interpolation steps

    Returns:
        attr:          [L, C] numpy array of attributions
    """
    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(window)
    attributions = ig.attribute(
        window,
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=False,
        internal_batch_size=1,
    )
    return attributions.squeeze(0).detach().cpu().numpy()  # [L, C]


# ── TimeSHAP — Monte Carlo Shapley Values ─────────────────────────────────────
# True Shapley values satisfy four axioms (efficiency, symmetry, dummy,
# additivity) that leave-one-out ablation does not.  We estimate them via the
# standard permutation-sampling approach:
#
#   φ(t) ≈ (1/n) Σ_π  [ v(S_π(t) ∪ {t}) − v(S_π(t)) ]
#
# where π is a uniformly sampled permutation of all L timesteps, S_π(t) is the
# set of timesteps that precede t in that permutation, and v(S) is the model
# score when only the timesteps in S contain real data (the rest are replaced by
# a feature-wise mean baseline).
#
# Each permutation requires L+1 forward passes.  All L+1 coalition tensors are
# batched into a single call (with an inner_batch_size guard against OOM).
# With n_background_samples=50 and a window length of 100, this is 50 batched
# calls of ≤64 windows each — roughly equivalent in wall time to the old LOO
# approach while producing theoretically grounded attributions.

def run_timeshap(wrapper, window, device, n_background_samples=50,
                 inner_batch_size=64):
    """
    Estimate per-timestep Shapley values via Monte Carlo permutation sampling.

    Algorithm
    ---------
    For each of n_background_samples random permutations of [0..L-1]:
      1. Start from an all-baseline tensor (feature-wise window mean).
      2. Reveal timesteps one by one in permutation order, each time replacing
         the baseline value at that index with the real observation.
      3. Record the marginal change in anomaly score when each timestep is added.
      4. Accumulate that marginal change into shapley_values[t].
    Divide by n_background_samples to get the Monte Carlo average.

    The baseline is the feature-wise mean of the window, which represents a
    "neutral" (non-anomalous) version of each feature and is broadcast across
    all timestep positions.

    Args:
        wrapper:              ATScoreWrapper or TranADScoreWrapper
        window:               [1, L, C] tensor — detached, already on device
        device:               torch device
        n_background_samples: number of random permutations to sample.
                              Higher values reduce variance (50–200 typical).
        inner_batch_size:     max coalition windows per forward pass.
                              Reduce if GPU OOM on long sequences.

    Returns:
        shapley_values: [L] float32 numpy array.
                        Positive  → timestep drives the score UP (toward anomaly).
                        Negative  → timestep drives the score DOWN (toward normal).
                        |φ(t)|    → magnitude of timestep's influence.
    """
    window = window.to(device)
    L = window.shape[1]

    # Feature-wise mean baseline: shape [1, L, C], same value at every timestep.
    # Using expand (not repeat) avoids an allocation; clone() before mutation.
    baseline = window.mean(dim=1, keepdim=True).expand_as(window)

    shapley_values = np.zeros(L, dtype=np.float64)

    for _ in range(n_background_samples):
        perm = np.random.permutation(L)

        # Build all L+1 coalition tensors for this permutation.
        # coalition[0]   = all-baseline  (empty coalition)
        # coalition[i+1] = baseline with the first i+1 timesteps of perm revealed
        coalitions = []
        current = baseline.clone()          # [1, L, C]
        coalitions.append(current.clone())

        for idx in perm:
            current = current.clone()       # avoid in-place aliasing
            current[0, idx, :] = window[0, idx, :]
            coalitions.append(current.clone())

        # Batch all L+1 coalitions along the batch dimension → [L+1, L, C]
        batch_tensor = torch.cat(coalitions, dim=0)

        # Evaluate in inner_batch_size chunks to respect GPU memory limits
        score_chunks = []
        for start in range(0, L + 1, inner_batch_size):
            chunk = batch_tensor[start : start + inner_batch_size]
            with torch.no_grad():
                score_chunks.append(wrapper(chunk).cpu())

        scores = torch.cat(score_chunks).numpy()    # [L+1]

        # Credit each timestep with its marginal contribution in this permutation
        for i, t in enumerate(perm):
            shapley_values[t] += scores[i + 1] - scores[i]

    shapley_values /= n_background_samples
    return shapley_values.astype(np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ig_heatmap(attr, feature_names, ticker, model_label, event_date, score_val, out_path):
    """Plot Integrated Gradients attribution as a heatmap."""
    L, C = attr.shape
    # Aggregate attribution magnitude per feature
    feat_importance = np.abs(attr).mean(axis=0)  # [C]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'Integrated Gradients — {ticker} | {model_label}\n'
        f'Event: {event_date} | Score: {score_val:.6f}',
        fontsize=14, fontweight='bold'
    )

    # Top: Full heatmap [L × C]
    ax1 = axes[0]
    # Use absolute attributions for visibility
    abs_attr = np.abs(attr).T  # [C, L]
    sns.heatmap(
        abs_attr, ax=ax1, cmap='YlOrRd',
        yticklabels=feature_names,
        xticklabels=False,
        cbar_kws={'label': '|Attribution|'}
    )
    ax1.set_xlabel('Timestep in Window')
    ax1.set_ylabel('Feature')
    ax1.set_title('Per-Timestep Feature Attribution')

    # Bottom: Bar chart of aggregated feature importance
    ax2 = axes[1]
    colors = sns.color_palette('YlOrRd', n_colors=C)
    sorted_idx = np.argsort(feat_importance)[::-1]
    ax2.barh(
        [feature_names[i] for i in sorted_idx],
        feat_importance[sorted_idx],
        color=[colors[min(i, C-1)] for i in range(C)]
    )
    ax2.set_xlabel('Mean |Attribution|')
    ax2.set_title('Aggregated Feature Importance')
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return feat_importance


def plot_timeshap(importance, ticker, model_label, event_date, score_val, out_path):
    """Plot TimeSHAP Shapley values as a bar chart."""
    L = len(importance)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(
        f'TimeSHAP — Temporal Shapley Values  |  {ticker} | {model_label}\n'
        f'Event: {event_date} | Score: {score_val:.6f}',
        fontsize=14, fontweight='bold'
    )

    timesteps = np.arange(L)
    colors = ['#d32f2f' if v > 0 else '#1976d2' for v in importance]
    ax.bar(timesteps, importance, color=colors, alpha=0.8, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Timestep in Window (oldest → newest)')
    ax.set_ylabel('Shapley Value  φ(t)')
    ax.set_title(
        'Per-Timestep Shapley Value  '
        '(red = pushes score toward anomaly, blue = toward normal)'
    )

    # Annotate top-5 most important timesteps
    top_k = min(5, L)
    top_idx = np.argsort(np.abs(importance))[-top_k:]
    for idx in top_idx:
        ax.annotate(
            f't={idx}',
            (idx, importance[idx]),
            textcoords='offset points', xytext=(0, 10 if importance[idx] > 0 else -15),
            ha='center', fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5)
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_bar(all_feat_importances, feature_names, out_path):
    """Plot a side-by-side comparison of feature importance between models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(feature_names))
    width = 0.35

    labels = list(all_feat_importances.keys())
    if len(labels) >= 2:
        ax.barh(x - width/2, all_feat_importances[labels[0]], width, label=labels[0], color='#1976d2', alpha=0.85)
        ax.barh(x + width/2, all_feat_importances[labels[1]], width, label=labels[1], color='#d32f2f', alpha=0.85)
    elif len(labels) == 1:
        ax.barh(x, all_feat_importances[labels[0]], width, label=labels[0], color='#1976d2', alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Mean |IG Attribution| (averaged across all events)')
    ax.set_title('Feature Importance Comparison: AT vs TranAD')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_timeshap_comparison(all_temporal, out_path):
    """Plot averaged Shapley value curves for both models.

    Curves are aligned to the shortest window length (min-length truncation)
    rather than zero-padded, which would artificially suppress the mean at
    early timesteps for models with longer windows.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {'AT': '#1976d2', 'TranAD': '#d32f2f'}
    for label, curves in all_temporal.items():
        if not curves:
            continue
        # Truncate to the shortest curve so no position is inflated by zeros
        min_len = min(len(c) for c in curves)
        truncated = np.stack([c[:min_len] for c in curves])  # [n_events, min_len]
        mean_curve = truncated.mean(axis=0)
        std_curve  = truncated.std(axis=0)

        x = np.arange(min_len)
        ax.plot(x, mean_curve, label=label,
                color=colors.get(label, 'gray'), linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.15, color=colors.get(label, 'gray'))

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Timestep in Window (oldest → newest)')
    ax.set_ylabel('Mean Shapley Value  φ(t)')
    ax.set_title(
        'TimeSHAP: Mean Temporal Shapley Values — AT vs TranAD\n'
        '(shaded band = ±1 std across events; curves truncated to shortest window)'
    )
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(event_results, out_path):
    """Generate a markdown summary of all XAI results."""
    lines = [
        '# XAI Analysis Report',
        '',
        'Explainable AI analysis of the top anomaly events using **Integrated Gradients** '
        'and **TimeSHAP** for the two selected models.',
        '',
        '---',
        '',
    ]

    for model_label, ticker_events in event_results.items():
        model_name = [m['name'] for m in MODELS if m['label'] == model_label][0]
        lines.append(f'## {model_name} (`{model_label}`)')
        lines.append('')

        for ticker, events in ticker_events.items():
            lines.append(f'### {ticker}')
            lines.append('')

            for ev in events:
                lines.append(f"#### {ev['date']} — score: {ev['score']:.6f}")
                lines.append('')

                # IG results
                lines.append('**Integrated Gradients — Top Features**')
                lines.append('')
                lines.append('| Rank | Feature | Mean |Attribution| |')
                lines.append('|------|---------|---------------------|')
                feat_imp = ev.get('feat_importance', {})
                sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
                for rank, (feat, val) in enumerate(sorted_feats, 1):
                    lines.append(f'| {rank} | {feat} | {val:.6f} |')
                lines.append('')

                if ev.get('ig_plot'):
                    lines.append(f"![IG Heatmap]({os.path.abspath(ev['ig_plot'])})")
                    lines.append('')

                # TimeSHAP results
                lines.append('**TimeSHAP — Temporal Shapley Values**')
                lines.append('')
                temporal = ev.get('temporal_importance', np.array([]))
                if len(temporal) > 0:
                    top5_idx = np.argsort(np.abs(temporal))[-5:][::-1]
                    lines.append('| Rank | Timestep | Shapley Value φ(t) |')
                    lines.append('|------|----------|--------------------|')
                    for rank, idx in enumerate(top5_idx, 1):
                        lines.append(f'| {rank} | t={idx} | {temporal[idx]:.6f} |')
                    lines.append('')
                    # Compute concentration: what % of total importance is in the last 20% of the window
                    window_len = len(temporal)
                    recent_cutoff = int(window_len * 0.8)
                    total_abs = np.abs(temporal).sum()
                    recent_abs = np.abs(temporal[recent_cutoff:]).sum()
                    recent_pct = (recent_abs / (total_abs + 1e-10)) * 100
                    lines.append(
                        f'> **Recency concentration**: {recent_pct:.1f}% of total |φ(t)| '
                        f'is concentrated in the last 20% of the window '
                        f'(timesteps {recent_cutoff}–{window_len-1}).'
                    )
                    lines.append('')

                if ev.get('ts_plot'):
                    lines.append(f"![TimeSHAP]({os.path.abspath(ev['ts_plot'])})")
                    lines.append('')

                lines.append('---')
                lines.append('')

    # Comparison section
    lines.append('## Cross-Model Comparison')
    lines.append('')
    lines.append('### Feature Importance')
    lines.append('')
    comparison_ig = os.path.join(OUT_DIR, 'comparison_feature_importance.png')
    if os.path.exists(comparison_ig):
        lines.append(f'![Feature Importance Comparison]({os.path.abspath(comparison_ig)})')
        lines.append('')

    lines.append('### Temporal Receptive Field')
    lines.append('')
    comparison_ts = os.path.join(OUT_DIR, 'comparison_temporal_importance.png')
    if os.path.exists(comparison_ts):
        lines.append(f'![Temporal Comparison]({os.path.abspath(comparison_ts)})')
        lines.append('')

    report_md = '\n'.join(lines)
    Path(out_path).write_text(report_md, encoding='utf-8')
    print(f"\nReport saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine feature count from data
    sample_ticker = TICKERS[0]
    data, dates, feature_names = load_ticker_data(sample_ticker)
    enc_in = data.shape[1]
    print(f"Features ({enc_in}): {feature_names}")

    # Collect results for report
    event_results = {}        # {model_label: {ticker: [event_dicts]}}
    all_feat_importances = {} # {model_label: [C] mean importance}
    all_temporal = {}         # {model_label: [list of [L] arrays]}

    for model_cfg in MODELS:
        label = model_cfg['label']
        det = model_cfg['detector']
        print(f"\n{'='*60}")
        print(f"Processing: {model_cfg['name']} ({label})")
        print(f"{'='*60}")

        if not os.path.exists(model_cfg['path']):
            print(f"  ✗ Weights not found: {model_cfg['path']} — skipping")
            continue

        model, config = load_model(model_cfg, enc_in, device)

        # Build score wrapper
        if det == 'Anomaly Transformer':
            wrapper = ATScoreWrapper(model).to(device)
            score_col = 'anomaly_score_AT'
        else:
            wrapper = TranADScoreWrapper(model).to(device)
            score_col = 'anomaly_score_TranAD'

        wrapper.eval()

        event_results[label] = {}
        all_feat_importances.setdefault(label, [])
        all_temporal.setdefault(label, [])

        for ticker in TICKERS:
            print(f"\n  Ticker: {ticker}")
            data, dates, feat_names = load_ticker_data(ticker)
            score_df = load_scores(ticker)

            top_events = get_top_events(score_df, score_col, TOP_N_EVENTS)
            event_results[label][ticker] = []

            for _, row in top_events.iterrows():
                event_date = str(row['Date'])[:10]
                score_val = float(row[score_col])
                original_idx = int(row['index'])

                print(f"    Event: {event_date} (score={score_val:.6f})")

                # Extract the window ending at (or containing) this event
                seq_len = config.seq_len
                n_rows = data.shape[0]

                # Map the original_idx from score_df back to the data tensor
                # Scores are end-aligned to data rows
                data_idx = original_idx
                if data_idx < seq_len:
                    # Not enough lookback, take the first full window
                    window_start = 0
                else:
                    window_start = data_idx - seq_len + 1

                window_end = window_start + seq_len
                if window_end > n_rows:
                    window_end = n_rows
                    window_start = window_end - seq_len

                window = data[window_start:window_end].unsqueeze(0).to(device)  # [1, L, C]
                window.requires_grad_(True)

                # ── Integrated Gradients ─────────────────────────
                print(f"      Running Integrated Gradients...")
                try:
                    ig_attr = run_integrated_gradients(wrapper, window, feat_names, n_steps=100)
                except Exception as e:
                    print(f"      IG failed: {e}")
                    ig_attr = np.zeros((seq_len, enc_in))

                ig_path = os.path.join(OUT_DIR, f'{ticker}_{label}_IG_{event_date}.png')
                feat_imp = plot_ig_heatmap(
                    ig_attr, feat_names, ticker, label, event_date, score_val, ig_path
                )
                print(f"      Saved IG heatmap → {ig_path}")

                # ── TimeSHAP ─────────────────────────────────────
                print(f"      Running TimeSHAP...")
                window_detached = data[window_start:window_end].unsqueeze(0).to(device)
                try:
                    temporal_imp = run_timeshap(wrapper, window_detached, device)
                except Exception as e:
                    print(f"      TimeSHAP failed: {e}")
                    temporal_imp = np.zeros(seq_len)

                ts_path = os.path.join(OUT_DIR, f'{ticker}_{label}_TimeSHAP_{event_date}.png')
                plot_timeshap(temporal_imp, ticker, label, event_date, score_val, ts_path)
                print(f"      Saved TimeSHAP plot → {ts_path}")

                # Collect results
                feat_imp_dict = {feat_names[i]: float(feat_imp[i]) for i in range(len(feat_names))}
                event_results[label][ticker].append({
                    'date': event_date,
                    'score': score_val,
                    'feat_importance': feat_imp_dict,
                    'temporal_importance': temporal_imp,
                    'ig_plot': ig_path,
                    'ts_plot': ts_path,
                })

                all_feat_importances[label].append(feat_imp)
                all_temporal[label].append(temporal_imp)

        # Cleanup GPU memory
        del model, wrapper
        torch.cuda.empty_cache() if device == 'cuda' else None

    # ── Cross-Model Comparison Plots ──────────────────────────────────────────
    print("\nGenerating comparison plots...")

    # Aggregate feature importance per model
    agg_feat = {}
    for label, imp_list in all_feat_importances.items():
        if imp_list:
            agg_feat[label] = np.mean(np.stack(imp_list), axis=0)
    if agg_feat:
        plot_comparison_bar(agg_feat, feature_names, os.path.join(OUT_DIR, 'comparison_feature_importance.png'))

    # Aggregate temporal importance
    if all_temporal:
        plot_timeshap_comparison(all_temporal, os.path.join(OUT_DIR, 'comparison_temporal_importance.png'))

    # ── Generate Report ──────────────────────────────────────────────────────
    generate_report(event_results, REPORT_PATH)

    print("\n✓ XAI analysis complete!")


if __name__ == '__main__':
    main()
