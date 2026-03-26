"""
plot_pipeline.py
----------------
Publication-quality anomaly score + price reconstruction plots
for the centralized pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_anomaly_scores(scores_df, ticker, models, out_dir):
    """Plot anomaly score time series for all detectors with anomaly bands.

    Args:
        scores_df: DataFrame with Date + score/prediction columns.
        ticker:    Ticker symbol.
        models:    List of model config dicts.
        out_dir:   Directory to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Parse dates
    df = scores_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    colors = {'AT': '#1976d2', 'TranAD': '#d32f2f'}

    for ax, model_cfg in zip(axes, models):
        label = model_cfg['label']
        score_col = model_cfg.get('score_col', f'anomaly_score_{label}')
        pred_col = model_cfg.get('pred_col', f'prediction_{label}')
        color = colors.get(label, '#333333')

        if score_col not in df.columns:
            ax.set_title(f'{model_cfg.get("name", label)} — No scores available')
            continue

        valid = df.dropna(subset=[score_col])
        ax.plot(valid['Date'], valid[score_col], color=color, linewidth=0.8, alpha=0.9)

        # Highlight predicted anomalies with translucent bands
        if pred_col in df.columns:
            pred_valid = valid.dropna(subset=[pred_col])
            anomaly_mask = pred_valid[pred_col] == 1
            if anomaly_mask.any():
                anomaly_dates = pred_valid.loc[anomaly_mask, 'Date']
                for dt in anomaly_dates:
                    ax.axvspan(
                        dt - pd.Timedelta(hours=12),
                        dt + pd.Timedelta(hours=12),
                        alpha=0.15, color='red', linewidth=0
                    )

        ax.set_ylabel('Anomaly Score')
        ax.set_title(f'{model_cfg.get("name", label)} — {ticker}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{ticker}_anomaly_scores.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved anomaly score plot → {out_path}")
    return out_path


def plot_price_reconstruction(scores_df, raw_df, ticker, models, out_dir):
    """Plot actual vs reconstructed close price.

    Args:
        scores_df: DataFrame with Date + recon_close columns.
        raw_df:    Raw OHLCV DataFrame with Date + Close.
        ticker:    Ticker symbol.
        models:    List of model config dicts.
        out_dir:   Directory to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = scores_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    raw = raw_df.copy()
    raw['Date'] = pd.to_datetime(raw['Date'])
    # Align to feature rows
    raw_tail = raw.iloc[-len(df):].reset_index(drop=True)
    raw_tail['Date'] = df['Date'].values

    fig, ax = plt.subplots(figsize=(16, 6))

    # Actual close
    ax.plot(raw_tail['Date'], raw_tail['Close'], color='black',
            linewidth=1.2, label='Actual Close', alpha=0.8)

    colors = {'AT': '#1976d2', 'TranAD': '#d32f2f'}
    for model_cfg in models:
        label = model_cfg['label']
        recon_col = f'recon_close_{label}'
        if recon_col in df.columns:
            valid = df.dropna(subset=[recon_col])
            ax.plot(valid['Date'], valid[recon_col],
                    color=colors.get(label, '#666'),
                    linewidth=0.8, alpha=0.7, linestyle='--',
                    label=f'{model_cfg.get("name", label)} Reconstruction')

    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price ($)')
    ax.set_title(f'{ticker} — Price Reconstruction Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{ticker}_price_reconstruction.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved price reconstruction plot → {out_path}")
    return out_path


def plot_combined(scores_df, raw_df, ticker, models, out_dir):
    """Combined anomaly score + price on a single figure.

    Args:
        scores_df: DataFrame with Date + score + recon columns.
        raw_df:    Raw OHLCV DataFrame.
        ticker:    Ticker symbol.
        models:    List of model config dicts.
        out_dir:   Directory to save plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = scores_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    raw = raw_df.copy()
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw_tail = raw.iloc[-len(df):].reset_index(drop=True)
    raw_tail['Date'] = df['Date'].values

    n_models = len(models)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(16, 4 * (n_models + 1)),
                             sharex=True,
                             gridspec_kw={'height_ratios': [2] + [1]*n_models})

    # Top panel: Price
    ax_price = axes[0]
    ax_price.plot(raw_tail['Date'], raw_tail['Close'], color='black',
                  linewidth=1.2, label='Close Price')
    ax_price.set_ylabel('Close ($)')
    ax_price.set_title(f'{ticker} — Anomaly Analysis Overview')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)

    # Bottom panels: Scores
    colors = {'AT': '#1976d2', 'TranAD': '#d32f2f'}
    for ax, model_cfg in zip(axes[1:], models):
        label = model_cfg['label']
        score_col = model_cfg.get('score_col', f'anomaly_score_{label}')
        pred_col = model_cfg.get('pred_col', f'prediction_{label}')
        color = colors.get(label, '#333')

        if score_col in df.columns:
            valid = df.dropna(subset=[score_col])
            ax.plot(valid['Date'], valid[score_col], color=color, linewidth=0.8)

            # Anomaly bands
            if pred_col in df.columns:
                pred_valid = valid.dropna(subset=[pred_col])
                anomaly_mask = pred_valid[pred_col] == 1
                if anomaly_mask.any():
                    for dt in pred_valid.loc[anomaly_mask, 'Date']:
                        # Also shade on the price panel
                        ax_price.axvspan(
                            dt - pd.Timedelta(hours=12),
                            dt + pd.Timedelta(hours=12),
                            alpha=0.1, color=color, linewidth=0
                        )
                        ax.axvspan(
                            dt - pd.Timedelta(hours=12),
                            dt + pd.Timedelta(hours=12),
                            alpha=0.15, color='red', linewidth=0
                        )

        ax.set_ylabel(f'{label} Score')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{ticker}_combined_analysis.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined plot → {out_path}")
    return out_path
