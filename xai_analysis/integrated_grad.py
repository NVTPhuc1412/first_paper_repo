"""
xai_analysis/integrated_grad.py
-------------------------------
Integrated Gradients attribution for the anomaly detection models.
Extracted from the original xai_analysis.py.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from captum.attr import IntegratedGradients
from model.utils import my_kl_loss


# ── Differentiable Score Wrappers ─────────────────────────────────────────────

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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ig_heatmap(attr, feature_names, ticker, model_label, event_date, score_val, out_path):
    """Plot Integrated Gradients attribution as a heatmap."""
    L, C = attr.shape
    feat_importance = np.abs(attr).mean(axis=0)  # [C]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f'Integrated Gradients — {ticker} | {model_label}\n'
        f'Event: {event_date} | Score: {score_val:.6f}',
        fontsize=14, fontweight='bold'
    )

    # Top: Full heatmap [L × C]
    ax1 = axes[0]
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


def plot_paper_ig(all_feat_importances, feature_names, out_path,
                  ticker=None, normalize=True):
    """Paper-style IG feature importance bar chart.

    Matches the style of ``ig_attribution.png``: horizontal bars, AT (blue) vs
    TranAD (green), percentage-normalised, sorted by pooled importance.

    Args:
        all_feat_importances: {label: np.array([C])}  — mean |IG| per feature.
        feature_names:        list of feature names.
        out_path:             Save path.
        ticker:               Ticker name for subtitle (optional).
        normalize:            If True, express as percentage of total |IG|.
    """
    labels = list(all_feat_importances.keys())
    colors = {'AT': '#4285F4', 'TranAD': '#34A853'}

    # Normalise to percentages
    imp = {}
    for lab in labels:
        arr = np.array(all_feat_importances[lab], dtype=np.float64)
        if normalize and arr.sum() > 0:
            arr = arr / arr.sum() * 100
        imp[lab] = arr

    # Sort features by pooled importance (descending)
    pooled = sum(imp[l] for l in labels) / len(labels)
    sort_idx = np.argsort(pooled)  # ascending → bottom to top

    sorted_names = [feature_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(feature_names))))

    y = np.arange(len(feature_names))
    bar_h = 0.35

    for i, lab in enumerate(labels):
        offset = -bar_h / 2 if i == 0 else bar_h / 2
        colour = colors.get(lab, '#999999')
        ax.barh(y + offset, imp[lab][sort_idx], bar_h,
                label=lab, color=colour, alpha=0.88, edgecolor='white', linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=10)
    unit = '(% of total |IG Attribution|)' if normalize else '(mean |IG Attribution|)'
    title = 'Feature Importance' + (f' — {ticker}' if ticker else ' (pooled, normalised)')
    ax.set_xlabel(f'Feature Importance {unit}', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='#CBD5E1')
    ax.spines[['top', 'right']].set_visible(False)

    if normalize:
        from matplotlib.ticker import PercentFormatter
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

