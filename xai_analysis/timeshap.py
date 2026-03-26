"""
xai_analysis/timeshap.py
------------------------
TimeSHAP — Monte Carlo Shapley value estimation for temporal importance.
Extracted from the original xai_analysis.py.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── TimeSHAP — Monte Carlo Shapley Values ─────────────────────────────────────
#
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

def run_timeshap(wrapper, window, device, n_background_samples=50,
                 inner_batch_size=64):
    """
    Estimate per-timestep Shapley values via Monte Carlo permutation sampling.

    Args:
        wrapper:              ATScoreWrapper or TranADScoreWrapper
        window:               [1, L, C] tensor — detached, already on device
        device:               torch device
        n_background_samples: number of random permutations to sample.
        inner_batch_size:     max coalition windows per forward pass.

    Returns:
        shapley_values: [L] float32 numpy array.
    """
    window = window.to(device)
    L = window.shape[1]

    # Feature-wise mean baseline
    baseline = window.mean(dim=1, keepdim=True).expand_as(window)

    shapley_values = np.zeros(L, dtype=np.float64)

    for _ in range(n_background_samples):
        perm = np.random.permutation(L)

        # Build all L+1 coalition tensors
        coalitions = []
        current = baseline.clone()
        coalitions.append(current.clone())

        for idx in perm:
            current = current.clone()
            current[0, idx, :] = window[0, idx, :]
            coalitions.append(current.clone())

        batch_tensor = torch.cat(coalitions, dim=0)

        # Evaluate in chunks
        score_chunks = []
        for start in range(0, L + 1, inner_batch_size):
            chunk = batch_tensor[start : start + inner_batch_size]
            with torch.no_grad():
                score_chunks.append(wrapper(chunk).cpu())

        scores = torch.cat(score_chunks).numpy()

        # Credit each timestep
        for i, t in enumerate(perm):
            shapley_values[t] += scores[i + 1] - scores[i]

    shapley_values /= n_background_samples
    return shapley_values.astype(np.float32)


# ── Plotting ──────────────────────────────────────────────────────────────────

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


def plot_timeshap_comparison(all_temporal, out_path):
    """Plot averaged Shapley value curves for both models.

    Curves are aligned to the shortest window length (min-length truncation).
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {'AT': '#1976d2', 'TranAD': '#d32f2f'}
    for label, curves in all_temporal.items():
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        truncated = np.stack([c[:min_len] for c in curves])
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


def plot_paper_timeshap_heatmap(all_temporal, out_path):
    """Paper-style side-by-side TimeSHAP heatmap.

    Matches ``timeshap_heatmap.png``: event × timestep grid, AT (Blues) on
    left, TranAD (Greens) on right, events sorted by anomaly score (low→high).

    Args:
        all_temporal: {label: list_of_shapley_arrays} — each array is [L].
        out_path:     Save path.
    """
    labels = list(all_temporal.keys())
    n_panels = len(labels)
    if n_panels == 0:
        return

    cmaps = {'AT': 'Blues', 'TranAD': 'Greens'}
    colors = {'AT': '#4285F4', 'TranAD': '#34A853'}

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, max(4, 0.25 * 30)))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        'Color = |Shapley value| · Events sorted by anomaly score (low → high)',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, label in zip(axes, labels):
        curves = all_temporal[label]
        if not curves:
            ax.set_title(f'{label} — no data')
            continue

        min_len = min(len(c) for c in curves)
        matrix = np.stack([np.abs(c[:min_len]) for c in curves])  # [n_events, L]

        n_events = matrix.shape[0]
        event_labels = [f'E{i+1}' for i in range(n_events)]

        cmap = cmaps.get(label, 'Blues')
        im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                       origin='lower', interpolation='nearest')
        ax.set_yticks(np.arange(n_events))
        ax.set_yticklabels(event_labels, fontsize=8)
        ax.set_xlabel(f'Timestep (0 = oldest, {min_len-1} = newest)', fontsize=10)

        color = colors.get(label, 'black')
        ax.set_title(label, fontsize=14, fontweight='bold', color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('|φ(t)|', fontsize=10)

    if n_panels > 0:
        axes[0].set_ylabel('Event (sorted by anomaly score)', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

