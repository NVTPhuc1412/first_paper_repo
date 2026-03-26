import os
import glob
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.config import Config
from model.main_model import Model
from model.scorer import AnomalyScorer

def _load_threshold(model_path, detector, threshold_path=None, fallback_percentile=98):
    """
    Load a threshold from a .npy file or search for one alongside the model.

    Search order:
      1. Explicit threshold_path if provided
      2. Auto-detect any *_thresholds.npy file next to the model .pth
      3. Try checkpoints/ directory for matching files
      4. Return None (caller will compute from scores)

    Args:
        model_path:          path to the .pth model file
        detector:            'Anomaly Transformer' or 'TranAD'
        threshold_path:      explicit path to a .npy threshold file (optional)
        fallback_percentile: percentile used by caller if no file is found

    Returns:
        threshold: np.ndarray (TranAD: [enc_in]) or float (AT), or None
    """
    candidates = []

    if threshold_path:
        candidates.append(threshold_path)

    # Auto-search next to the model file
    model_dir = os.path.dirname(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    candidates += glob.glob(os.path.join(model_dir, f"{model_stem}*thresholds.npy"))

    # Look in checkpoints/
    project_root = os.path.dirname(os.path.abspath(model_path))
    enc_label = 'None' if 'Anomaly Transformer' in model_path else model_stem.split('_')[0]
    det_label = 'Anomaly Transformer' if 'Anomaly Transformer' in model_path else 'TranAD'
    ckpt_pattern = os.path.join(
        project_root, 'checkpoints',
        f"{enc_label}-{det_label}_epochs",
        '*_thresholds.npy'
    )
    ckpt_files = sorted(glob.glob(ckpt_pattern))
    if ckpt_files:
        # Use the latest epoch's threshold file
        candidates.append(ckpt_files[-1])

    for cpath in candidates:
        if cpath and os.path.exists(cpath):
            loaded = np.load(cpath)
            if detector == 'TranAD':
                print(f"  Loaded thresholds ({os.path.basename(cpath)}): per-feature {np.round(loaded, 6)}")
                return loaded          # [enc_in]
            else:
                val = float(loaded[0])
                print(f"  Loaded threshold ({os.path.basename(cpath)}): {val:.6f}")
                return val

    print(f"  No threshold file found — will use {fallback_percentile}th-percentile of test scores.")
    return None


def _apply_threshold(seq_scores, detector, threshold):
    """
    Convert per-timestep scores to binary predictions, mirroring tester.py logic.

    AT:     scalar threshold  → pred[t] = (score[t] > threshold)
    TranAD: per-feature thresholds → mean(score) compared to mean(thresholds)

    Args:
        seq_scores: np.ndarray  AT: [T]  | TranAD: [T, enc_in]
        detector:   str
        threshold:  float (AT) or np.ndarray [enc_in] (TranAD)

    Returns:
        pred: np.ndarray [T] of int (0/1)
    """
    if detector == 'TranAD':
        mean_score = seq_scores.mean(axis=-1)       # [T]
        mean_thresh = float(np.mean(threshold))
        return (mean_score >= mean_thresh).astype(int)
    else:
        return (seq_scores > threshold).astype(int)


def _aggregate_recon(window_recons, detector, test_len, seq_len):
    """
    Aggregate per-window reconstruction tensors into a per-timestep sequence.

    AT:     non-overlapping windows, each [seq_len, enc_in]
            → concatenate directly → [test_len, enc_in]
    TranAD: stride-1 windows, each contributes the last timestep [1, enc_in]
            → stack last steps → [test_len, enc_in]

    Args:
        window_recons: np.ndarray
            AT:     [n_windows, seq_len, enc_in]
            TranAD: [n_windows, 1, enc_in]
        detector:  str
        test_len:  int — expected output length
        seq_len:   int

    Returns:
        seq_recon: np.ndarray [test_len, enc_in]
    """
    if detector == 'Anomaly Transformer':
        # [n_windows, seq_len, enc_in] → [n_windows*seq_len, enc_in]
        seq = window_recons.reshape(-1, window_recons.shape[-1])
    else:
        # TranAD: [n_windows, 1, enc_in] → [n_windows, enc_in]
        seq = window_recons.squeeze(1)

    if len(seq) < test_len:
        pad_len = test_len - len(seq)
        pad_val = seq.mean(axis=0, keepdims=True)
        seq = np.concatenate([seq, np.repeat(pad_val, pad_len, axis=0)], axis=0)
    return seq[-test_len:]  # [test_len, enc_in]


def _recon_to_close(recon_seq, scaler, feature_names, raw_close_series, anchor_idx):
    """
    Convert a scaled reconstruction sequence back to a Close price curve.

    Steps:
      1. Inverse-transform the scaled reconstruction to raw feature space.
      2. Extract the reconstructed f1_log_ret column.
      3. Compute Cumulative Close price anchored to the real Close at anchor_idx:
         Close_recon[t] = Close_anchor * exp(cumsum(log_ret_recon[1..t]))

    Args:
        recon_seq:        np.ndarray [T, enc_in] — scaled reconstruction
        scaler:           fitted RobustScaler for this ticker
        feature_names:    list of str — column names matching scaler features
        raw_close_series: pd.Series — full raw Close prices aligned to feature rows
        anchor_idx:       int — index in raw_close_series of the timestep just
                          before the reconstruction window starts

    Returns:
        close_recon: np.ndarray [T] — reconstructed Close prices
        log_ret_recon: np.ndarray [T] — reconstructed log returns
    """
    # Inverse-scale the full feature vector
    recon_raw = scaler.inverse_transform(recon_seq)  # [T, enc_in]

    # Find the f1_log_ret column
    if 'f1_log_ret' not in feature_names:
        raise ValueError("f1_log_ret not found in feature_names — cannot recover Close price.")
    log_ret_idx = feature_names.index('f1_log_ret')
    log_ret = recon_raw[:, log_ret_idx]              # [T]

    # Reconstruct close price cumulatively from the anchor point
    # anchor_idx is the index just before the first timestep of recon_seq
    anchor_close = float(raw_close_series.iloc[anchor_idx])
    close_recon = anchor_close * np.exp(np.cumsum(log_ret))
    return close_recon, log_ret


def get_scores_for_model(
    model_name, model_path, test_dir, encoder, detector, target_tickers,
    batch_size=256, threshold_path=None, fallback_percentile=98
):
    """
    Extract per-timestep anomaly scores and binary predictions for the given
    model and target tickers.

    Args:
        model_name:          short label (e.g. 'AT')
        model_path:          path to the saved .pth weights
        test_dir:            directory containing per-ticker feature CSVs
        encoder:             encoder key (e.g. 'TimesNet') or None
        detector:            'Anomaly Transformer' or 'TranAD'
        target_tickers:      list of ticker symbols to process
        batch_size:          DataLoader batch size
        threshold_path:      optional explicit path to a .npy threshold file
        fallback_percentile: percentile threshold used when no .npy file found

    Returns:
        ticker_seq_scores:  list of (ticker_name, scores_array) — scores per ticker
        ticker_predictions: list of (ticker_name, pred_array) — binary preds per ticker
        ticker_recons:      list of (ticker_name, recon_array) — scaled recon [T, enc_in]
        test_start:         int — first index of the test window inside the full sequence
        test_len:           int — number of test timesteps per ticker
        value_cols:         list of feature column names
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Inspect data to find enc_in
    files = glob.glob(os.path.join(test_dir, '*.csv'))
    # Filter files for target_tickers only to speed up
    files = [f for f in files if any(t in os.path.basename(f) for t in target_tickers)]
    if not files:
        raise ValueError(f"No target tickers found in {test_dir}")

    df_sample = pd.read_csv(files[0])
    value_cols = [c for c in df_sample.columns if c not in ['Date', 'ticker', 'anomaly'] and not c.startswith('Is_Anomaly')]
    enc_in = len(value_cols)
    print(f"[{detector}] Using features: {value_cols}")

    config = Config(
        encoder=encoder,
        detector=detector,
        enc_in=enc_in,
        batch_size=batch_size
    )
    config.device = device

    # We use get_data_loaders but we cannot use is_test=True because analysis data lacks 'anomaly' column
    # Let's read the data manually like dataset_n_dataloader does, but skipping labels
    from model.dataset_n_dataloader import StockDataset

    ticker_names = [os.path.basename(path).replace('.csv', '') for path in files]
    n_tickers = len(files)
    n_rows = len(df_sample)
    data_tensor = torch.empty(n_tickers, n_rows, enc_in, dtype=torch.float32)
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        data_tensor[i] = torch.from_numpy(df[value_cols].to_numpy().astype(np.float32))

    # We need to compute test_start and test_len to mimic tester
    if detector == 'Anomaly Transformer':
        test_overlap = False
        test_start = n_rows % config.seq_len
        test_len = n_rows - test_start
    elif detector == 'TranAD':
        test_overlap = True
        test_start = 0
        test_len = n_rows - config.seq_len + 1
    else:
        test_overlap = True
        test_start = 0
        test_len = n_rows

    labels_tensor = torch.zeros(n_tickers, n_rows, dtype=torch.float32)
    test_ds = StockDataset(
        config, data_tensor, labels_tensor=labels_tensor,
        split='test', overlap=test_overlap
    )
    # The actual test_len computed by StockDataset:
    assert test_ds.test_len == test_len

    from torch.utils.data import DataLoader
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False)

    # Load model
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scorer = AnomalyScorer(config)

    # Collect window scores AND reconstructions in one forward pass
    all_scores = []
    all_recons = []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Scoring {detector}"):
            batch = batch.to(device)
            outputs = model(batch)
            scores = scorer.score_batch(batch, outputs)
            all_scores.append(scores)

            # Extract reconstruction from model outputs (already denormed by Model)
            if detector == 'Anomaly Transformer':
                # outputs = (enc_out, series, prior, sigmas)
                # enc_out: [B, L, enc_in] — full window reconstruction
                recon = outputs[0].cpu().numpy()       # [B, seq_len, enc_in]
            else:
                # TranAD: outputs = (x1, x2)
                # x1, x2: [B, 1, enc_in] — reconstruction of last timestep
                x1, x2 = outputs[0], outputs[1]
                recon = (0.5 * x1.cpu() + 0.5 * x2.cpu()).numpy()  # [B, 1, enc_in]
            all_recons.append(recon)

    window_scores = np.concatenate(all_scores, axis=0)
    window_recons = np.concatenate(all_recons, axis=0)  # [total_windows, ..., enc_in]

    # Reshape scores and reconstructions to per-ticker sequences
    ticker_seq_scores = []
    ticker_recons = []
    for t in range(n_tickers):
        start = t * test_ds.n_per_ticker
        end = start + test_ds.n_per_ticker

        windows_t = window_scores[start:end]
        seq_t = scorer.aggregate_to_sequence(windows_t, test_len)
        ticker_seq_scores.append((ticker_names[t], seq_t))

        recon_windows_t = window_recons[start:end]       # [n_per_ticker, ..., enc_in]
        recon_seq_t = _aggregate_recon(recon_windows_t, detector, test_len, config.seq_len)
        ticker_recons.append((ticker_names[t], recon_seq_t))

    # ── Threshold & predictions ───────────────────────────────────────────────
    print(f"[{detector}] Resolving threshold...")
    threshold = _load_threshold(model_path, detector, threshold_path, fallback_percentile)

    if threshold is None:
        # Compute from all test scores concatenated across tickers
        all_seq = np.concatenate([s for _, s in ticker_seq_scores], axis=0)  # [T_total, ...]
        if detector == 'TranAD':
            # per-feature percentile
            threshold = np.percentile(all_seq.reshape(-1, all_seq.shape[-1]),
                                      fallback_percentile, axis=0)  # [enc_in]
            print(f"  Computed {fallback_percentile}th-pct thresholds per feature: {np.round(threshold, 6)}")
        else:
            threshold = float(np.percentile(all_seq, fallback_percentile))
            print(f"  Computed {fallback_percentile}th-pct threshold: {threshold:.6f}")

    ticker_predictions = [
        (name, _apply_threshold(seq, detector, threshold))
        for name, seq in ticker_seq_scores
    ]

    return ticker_seq_scores, ticker_predictions, ticker_recons, test_start, test_len, value_cols

def run_scoring_pipeline(
    ticker, data_dir, raw_data_dir, scaler_path, out_dir,
    models, batch_size=256,
):
    """Run anomaly scoring for a single ticker with all configured models.

    This is the public API used by pipeline.py.

    Args:
        ticker:       Ticker symbol (e.g. 'NVDA').
        data_dir:     Directory containing the scaled feature CSV.
        raw_data_dir: Directory containing raw OHLCV CSVs (for dates/close).
        scaler_path:  Path to the scaler .joblib file for this ticker.
        out_dir:      Directory to save scores CSV.
        models:       List of model config dicts (from pipeline_config).
        batch_size:   DataLoader batch size.

    Returns:
        scores_df: pd.DataFrame with Date + score/prediction columns for each model.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load scaler
    if os.path.isdir(scaler_path):
        # scaler_path is a directory — look for the ticker's scaler
        import glob as _glob
        candidates = _glob.glob(os.path.join(str(scaler_path), f"*{ticker}*scaler*.joblib"))
        if candidates:
            scaler_dict = {ticker: joblib.load(candidates[0])}
        else:
            scaler_dict = {}
    elif os.path.isfile(scaler_path):
        loaded = joblib.load(scaler_path)
        if isinstance(loaded, dict):
            scaler_dict = loaded
        else:
            scaler_dict = {ticker: loaded}
    else:
        scaler_dict = {}

    # Read feature CSV to get dates and row count
    ticker_csv = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(ticker_csv):
        raise FileNotFoundError(f"Scaled feature CSV not found: {ticker_csv}")

    df_features = pd.read_csv(ticker_csv)
    n_rows = len(df_features)

    # Get dates from raw data
    raw_csv = os.path.join(raw_data_dir, f"{ticker}.csv")
    if os.path.exists(raw_csv):
        raw_df = pd.read_csv(raw_csv, usecols=['Date', 'Close'])
        raw_df_tail = raw_df.iloc[-n_rows:].reset_index(drop=True)
        dates = raw_df_tail['Date']
        raw_close = raw_df_tail['Close']
    else:
        dates = pd.Series(range(n_rows), name='Date')
        raw_close = None

    # Build the result DataFrame
    result_df = pd.DataFrame({'Date': dates})

    value_cols = [c for c in df_features.columns
                  if c not in ['Date', 'ticker', 'anomaly']
                  and not c.startswith('Is_Anomaly')]

    for model_cfg in models:
        label = model_cfg['label']
        detector = model_cfg['detector']
        encoder = model_cfg['encoder']
        model_path = model_cfg['path'] if 'path' in model_cfg else str(model_cfg.get('weights', ''))
        threshold_path_val = model_cfg.get('threshold_path', None)

        if not os.path.exists(model_path):
            print(f"  ✗ Weights not found: {model_path} — skipping {label}")
            continue

        print(f"\n  Scoring with {model_cfg.get('name', label)}...")
        try:
            scores, preds, recons, test_start, test_len, cols = get_scores_for_model(
                label, model_path, data_dir,
                encoder=encoder, detector=detector,
                target_tickers=[ticker],
                batch_size=batch_size,
                threshold_path=threshold_path_val,
            )

            # Unpack single-ticker results
            score_arr = dict(scores).get(ticker)
            pred_arr = dict(preds).get(ticker)
            recon_arr = dict(recons).get(ticker)

            score_col = model_cfg.get('score_col', f'anomaly_score_{label}')
            pred_col = model_cfg.get('pred_col', f'prediction_{label}')

            # Initialize columns
            result_df[score_col] = np.nan
            result_df[pred_col] = np.nan
            result_df[f'recon_close_{label}'] = np.nan
            result_df[f'recon_log_ret_{label}'] = np.nan

            if score_arr is not None:
                if detector == 'TranAD' and score_arr.ndim > 1:
                    mean_scores = score_arr.mean(axis=-1)
                else:
                    mean_scores = score_arr
                result_df.iloc[-test_len:, result_df.columns.get_loc(score_col)] = mean_scores

            if pred_arr is not None:
                result_df.iloc[-test_len:, result_df.columns.get_loc(pred_col)] = pred_arr

            if recon_arr is not None and raw_close is not None and ticker in scaler_dict:
                anchor_row = n_rows - test_len - 1
                anchor_row = max(anchor_row, 0)
                try:
                    close_recon, log_ret_recon = _recon_to_close(
                        recon_arr, scaler_dict[ticker], cols, raw_close, anchor_row
                    )
                    result_df.iloc[-test_len:, result_df.columns.get_loc(f'recon_close_{label}')] = close_recon
                    result_df.iloc[-test_len:, result_df.columns.get_loc(f'recon_log_ret_{label}')] = log_ret_recon
                except Exception as e:
                    print(f"    Warning: Close reconstruction failed for {ticker}/{label}: {e}")

        except Exception as e:
            print(f"    Error scoring {label}: {e}")
            import traceback
            traceback.print_exc()

    # Save scores CSV
    out_csv = os.path.join(out_dir, f"{ticker}_scores.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\n  Saved scores → {out_csv} ({n_rows} rows)")

    return result_df


def main():

    target_tickers = ['NVDA', 'TSLA', 'INTC']
    test_dir = './data/engineered_analysis_data_stat'
    if not os.path.exists(test_dir):
        test_dir = './data/engineered_analysis_data'
    print(f"Using data directory: {test_dir}")

    model1_path = './best_saved_model/None_Anomaly Transformer.pth'
    threshold_path1 = './best_saved_model/None_Anomaly Transformer_percentile_thresholds.npy'
    model2_path = './best_saved_model/TimesNet_TranAD.pth'
    threshold_path2 = './best_saved_model/TimesNet_TranAD_percentile_thresholds.npy'

    out_dir = './data/extracted_scores'
    os.makedirs(out_dir, exist_ok=True)

    # ── Load scaler (needed to inverse-transform model reconstructions) ───
    scaler_path = './utils/scalers_stat/analysis_scalers.joblib'
    if not os.path.exists(scaler_path):
        scaler_path = './utils/analysis_scalers.joblib'
    scalers = joblib.load(scaler_path)
    print(f"Loaded scalers from {scaler_path} (tickers: {list(scalers.keys())})")

    raw_data_dir = './data/raw_tickers_data'

    print("Extracting AT scores...")
    at_scores, at_preds, at_recons, at_start, at_len, at_cols = get_scores_for_model(
        'AT', model1_path, test_dir, encoder=None, detector='Anomaly Transformer', target_tickers=target_tickers,
        threshold_path=threshold_path1
    )

    print("Extracting TranAD scores...")
    tranad_scores, tranad_preds, tranad_recons, tranad_start, tranad_len, tranad_cols = get_scores_for_model(
        'TranAD', model2_path, test_dir, encoder='TimesNet', detector='TranAD', target_tickers=target_tickers,
        threshold_path=threshold_path2
    )

    at_dict = dict(at_scores)
    at_pred_dict = dict(at_preds)
    at_recon_dict = dict(at_recons)
    tranad_dict = dict(tranad_scores)
    tranad_pred_dict = dict(tranad_preds)
    tranad_recon_dict = dict(tranad_recons)

    for ticker in target_tickers:
        ticker_csv = os.path.join(test_dir, f"{ticker}.csv")
        if not os.path.exists(ticker_csv):
            print(f"Warning: {ticker} not found in {test_dir}")
            continue

        df = pd.read_csv(ticker_csv)
        n_rows = len(df)

        # ── Attach dates from raw data (end-aligned) ─────────────────────
        raw_csv = os.path.join(raw_data_dir, f"{ticker}.csv")
        if os.path.exists(raw_csv):
            raw_df = pd.read_csv(raw_csv, usecols=['Date', 'Close'])
            # Engineered data keeps the last n_rows rows of raw data
            raw_df_tail = raw_df.iloc[-n_rows:].reset_index(drop=True)
            dates = raw_df_tail['Date']
            raw_close = raw_df_tail['Close']          # aligned to feature rows
            raw_close_full = raw_df['Close']           # full series for anchor lookup
        else:
            print(f"Warning: raw CSV not found for {ticker}, dates will be integer index")
            dates = pd.Series(range(n_rows), name='Date')
            raw_close = None
            raw_close_full = None

        df.insert(0, 'Date', dates)

        # Initialize score, prediction, and reconstruction columns
        df['anomaly_score_AT'] = np.nan
        df['prediction_AT'] = np.nan
        df['recon_close_AT'] = np.nan
        df['recon_log_ret_AT'] = np.nan
        df['anomaly_score_TranAD'] = np.nan
        df['prediction_TranAD'] = np.nan
        df['recon_close_TranAD'] = np.nan
        df['recon_log_ret_TranAD'] = np.nan

        # ── AT ────────────────────────────────────────────────────────────
        if ticker in at_dict:
            scores = at_dict[ticker]               # [at_len]
            df.iloc[-at_len:, df.columns.get_loc('anomaly_score_AT')] = scores
        if ticker in at_pred_dict:
            preds = at_pred_dict[ticker]           # [at_len]
            df.iloc[-at_len:, df.columns.get_loc('prediction_AT')] = preds
        if ticker in at_recon_dict and raw_close is not None and ticker in scalers:
            recon_seq = at_recon_dict[ticker]      # [at_len, enc_in]
            # Anchor: the real Close at the row just before the AT test window
            # AT skips test_start rows, so anchor = test_start - 1 in the FEATURE space
            # which aligns to row (n_rows - at_len - 1) of raw_close
            anchor_row = n_rows - at_len - 1
            anchor_row = max(anchor_row, 0)
            try:
                close_recon, log_ret_recon = _recon_to_close(
                    recon_seq, scalers[ticker], at_cols, raw_close, anchor_row
                )
                df.iloc[-at_len:, df.columns.get_loc('recon_close_AT')] = close_recon
                df.iloc[-at_len:, df.columns.get_loc('recon_log_ret_AT')] = log_ret_recon

            except Exception as e:
                print(f"  Warning: AT close reconstruction failed for {ticker}: {e}")

        # ── TranAD ────────────────────────────────────────────────────────
        if ticker in tranad_dict:
            scores = tranad_dict[ticker]           # [tranad_len, enc_in]
            mean_scores = scores.mean(axis=-1)
            df.iloc[-tranad_len:, df.columns.get_loc('anomaly_score_TranAD')] = mean_scores
        if ticker in tranad_pred_dict:
            preds = tranad_pred_dict[ticker]       # [tranad_len]
            df.iloc[-tranad_len:, df.columns.get_loc('prediction_TranAD')] = preds
        if ticker in tranad_recon_dict and raw_close is not None and ticker in scalers:
            recon_seq = tranad_recon_dict[ticker]  # [tranad_len, enc_in]
            # TranAD covers nearly the whole sequence (seq_len warm-up only)
            anchor_row = n_rows - tranad_len - 1
            anchor_row = max(anchor_row, 0)
            try:
                close_recon, log_ret_recon = _recon_to_close(
                    recon_seq, scalers[ticker], tranad_cols, raw_close, anchor_row
                )
                df.iloc[-tranad_len:, df.columns.get_loc('recon_close_TranAD')] = close_recon
                df.iloc[-tranad_len:, df.columns.get_loc('recon_log_ret_TranAD')] = log_ret_recon

            except Exception as e:
                print(f"  Warning: TranAD close reconstruction failed for {ticker}: {e}")

        out_csv = os.path.join(out_dir, f"{ticker}_scores.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved extracted scores to {out_csv} ({n_rows} rows, dates: {dates.iloc[0]} → {dates.iloc[-1]})")

if __name__ == '__main__':
    main()
