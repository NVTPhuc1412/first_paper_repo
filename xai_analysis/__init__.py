"""
xai_analysis
=============
Explainable AI package for the anomaly detection pipeline.

Modules:
    integrated_grad  — Integrated Gradients (Captum)
    timeshap         — TimeSHAP (Monte Carlo Shapley values)
    news_fetcher     — Lightweight GDELT headline fetcher
    attribute        — LLM-powered anomaly attribution
"""

from xai_analysis.integrated_grad import (
    ATScoreWrapper,
    TranADScoreWrapper,
    run_integrated_gradients,
    plot_ig_heatmap,
    plot_comparison_bar,
    plot_paper_ig,
)
from xai_analysis.timeshap import (
    run_timeshap,
    plot_timeshap,
    plot_timeshap_comparison,
    plot_paper_timeshap_heatmap,
)


def run_xai_for_ticker(
    ticker, scores_df, data_dir, raw_data_dir, out_dir, cache_dir,
    models, top_n=5, ig_n_steps=100, timeshap_n_samples=50,
):
    """Run Integrated Gradients and TimeSHAP on the top anomaly events.

    This is the main XAI entry point used by pipeline.py.

    Args:
        ticker:             Ticker symbol.
        scores_df:          DataFrame with Date + score columns.
        data_dir:           Directory with scaled feature CSVs.
        raw_data_dir:       Directory with raw OHLCV CSVs.
        out_dir:            Directory for XAI plots.
        cache_dir:          Directory for caching numerical results (.npz).
        models:             List of model config dicts.
        top_n:              Number of top anomaly events to analyze per model.
        ig_n_steps:         IG interpolation steps.
        timeshap_n_samples: Number of permutation samples for TimeSHAP.

    Returns:
        event_results: dict {model_label: [event_dicts]}
    """
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import torch

    from model.config import Config
    from model.main_model import Model

    warnings.filterwarnings('ignore', category=FutureWarning)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load ticker data
    csv_path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(csv_path)
    value_cols = [
        c for c in df.columns
        if c not in ['Date', 'ticker', 'anomaly'] and not c.startswith('Is_Anomaly')
    ]
    enc_in = len(value_cols)
    data = torch.from_numpy(df[value_cols].to_numpy().astype(np.float32))

    # Load dates from raw data
    raw_csv = os.path.join(raw_data_dir, f'{ticker}.csv')
    if os.path.exists(raw_csv):
        raw_df = pd.read_csv(raw_csv, usecols=['Date'])
        dates = raw_df['Date'].iloc[-len(df):].reset_index(drop=True)
    else:
        dates = pd.Series(range(len(df)), name='Date')

    event_results = {}
    all_feat_importances = {}
    all_temporal = {}

    for model_cfg in models:
        label = model_cfg['label']
        det = model_cfg['detector']
        encoder = model_cfg.get('encoder')
        model_path = model_cfg['path'] if 'path' in model_cfg else str(model_cfg.get('weights', ''))
        score_col = model_cfg.get('score_col', f'anomaly_score_{label}')

        if not os.path.exists(model_path):
            print(f"  ✗ Weights not found: {model_path} — skipping {label}")
            continue

        print(f"\n  XAI analysis for {model_cfg.get('name', label)}...")

        # Load model
        config = Config(encoder=encoder, detector=det, enc_in=enc_in)
        config.device = device
        model = Model(config).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Build score wrapper
        if det == 'Anomaly Transformer':
            wrapper = ATScoreWrapper(model).to(device)
        else:
            wrapper = TranADScoreWrapper(model).to(device)
        wrapper.eval()

        # Get top-N events
        if score_col not in scores_df.columns:
            print(f"    Score column {score_col} not found — skipping")
            continue

        valid = scores_df.dropna(subset=[score_col])
        top_events = valid.nlargest(top_n, score_col)

        event_results[label] = []
        all_feat_importances.setdefault(label, [])
        all_temporal.setdefault(label, [])

        for _, row in top_events.iterrows():
            event_date = str(row['Date'])[:10]
            score_val = float(row[score_col])
            original_idx = int(row.name) if hasattr(row, 'name') else 0

            # Try to find the index in scores_df
            try:
                original_idx = scores_df.index[scores_df['Date'] == row['Date']][0]
            except (IndexError, KeyError):
                original_idx = int(valid.index[valid[score_col] == score_val][0])

            print(f"    Event: {event_date} (score={score_val:.6f})")

            seq_len = config.seq_len
            n_rows = data.shape[0]
            data_idx = original_idx
            if data_idx < seq_len:
                window_start = 0
            else:
                window_start = data_idx - seq_len + 1
            window_end = window_start + seq_len
            if window_end > n_rows:
                window_end = n_rows
                window_start = window_end - seq_len

            # Check cache
            ig_cache_file = os.path.join(cache_dir, f'{label}_{event_date}_ig.npz')
            ts_cache_file = os.path.join(cache_dir, f'{label}_{event_date}_timeshap.npz')

            # ── Integrated Gradients ──────────────────────────
            if os.path.exists(ig_cache_file):
                print(f"      Loading cached IG results...")
                cached = np.load(ig_cache_file)
                ig_attr = cached['attributions']
                feat_imp = cached['feature_importance']
            else:
                print(f"      Running Integrated Gradients...")
                window = data[window_start:window_end].unsqueeze(0).to(device)
                window.requires_grad_(True)
                try:
                    ig_attr = run_integrated_gradients(
                        wrapper, window, value_cols, n_steps=ig_n_steps
                    )
                    feat_imp = np.abs(ig_attr).mean(axis=0)
                except Exception as e:
                    print(f"      IG failed: {e}")
                    ig_attr = np.zeros((seq_len, enc_in))
                    feat_imp = np.zeros(enc_in)

                # Cache results
                np.savez_compressed(ig_cache_file,
                                    attributions=ig_attr,
                                    feature_importance=feat_imp,
                                    feature_names=value_cols)

            ig_plot_path = os.path.join(out_dir, f'{ticker}_{label}_IG_{event_date}.png')
            plot_ig_heatmap(
                ig_attr, value_cols, ticker, label, event_date, score_val, ig_plot_path
            )

            # ── TimeSHAP ──────────────────────────────────────
            if os.path.exists(ts_cache_file):
                print(f"      Loading cached TimeSHAP results...")
                cached = np.load(ts_cache_file)
                temporal_imp = cached['shapley_values']
            else:
                print(f"      Running TimeSHAP...")
                window_detached = data[window_start:window_end].unsqueeze(0).to(device)
                try:
                    temporal_imp = run_timeshap(
                        wrapper, window_detached, device,
                        n_background_samples=timeshap_n_samples
                    )
                except Exception as e:
                    print(f"      TimeSHAP failed: {e}")
                    temporal_imp = np.zeros(seq_len)

                # Cache results
                np.savez_compressed(ts_cache_file, shapley_values=temporal_imp)

            ts_plot_path = os.path.join(out_dir, f'{ticker}_{label}_TimeSHAP_{event_date}.png')
            plot_timeshap(temporal_imp, ticker, label, event_date, score_val, ts_plot_path)

            # Collect results
            feat_imp_dict = {value_cols[i]: float(feat_imp[i]) for i in range(len(value_cols))}
            event_results[label].append({
                'date': event_date,
                'score': score_val,
                'feat_importance': feat_imp_dict,
                'temporal_importance': temporal_imp,
                'ig_plot': ig_plot_path,
                'ts_plot': ts_plot_path,
            })
            all_feat_importances[label].append(feat_imp)
            all_temporal[label].append(temporal_imp)

        # Cleanup
        del model, wrapper
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Cross-model comparison plots
    if len(all_feat_importances) > 0:
        agg_feat = {}
        for label, imp_list in all_feat_importances.items():
            if imp_list:
                agg_feat[label] = np.mean(np.stack(imp_list), axis=0)
        if agg_feat:
            plot_comparison_bar(
                agg_feat, value_cols,
                os.path.join(out_dir, 'comparison_feature_importance.png')
            )
            # Paper-style IG bar chart
            plot_paper_ig(
                agg_feat, value_cols,
                os.path.join(out_dir, f'{ticker}_ig_paper.png'),
                ticker=ticker
            )

    if len(all_temporal) > 0:
        plot_timeshap_comparison(
            all_temporal,
            os.path.join(out_dir, 'comparison_temporal_importance.png')
        )
        # Paper-style TimeSHAP heatmap
        plot_paper_timeshap_heatmap(
            all_temporal,
            os.path.join(out_dir, f'{ticker}_timeshap_paper.png')
        )

    return event_results
