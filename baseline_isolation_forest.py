"""
baseline_isolation_forest.py
─────────────────────────────
Trains and evaluates an Isolation Forest baseline on the engineered
stock-market anomaly dataset.

Dataset schema (one CSV per ticker)
─────────────────────────────────────
Features (8 columns, already normalised):
    f1_log_ret, f2_gap_ret, f3_parkinson, f4_vol_z,
    f6_rsi_14,  f7_macd_hist, f8_bb_width, f5_rel_ret

Label columns (bool):
    anomaly               – overall label (OR of the three below)
    Is_Anomaly_Point      – isolated single-day anomaly
    Is_Anomaly_Contextual – short-run anomaly surrounded by normals
    Is_Anomaly_Collective – long-run or boundary anomaly

Train directory  – CSVs are loaded; label columns are ignored.
Test  directory  – CSVs must contain label columns.

Threshold strategies
─────────────────────
  pot        – Peaks-Over-Threshold (EVT / Generalised Pareto) on
               the training anomaly-score distribution.
  percentile – Fixed high-percentile cut on training scores.

Anomaly-state adjustment (point-adjust / PA)
─────────────────────────────────────────────
If any point in a labelled anomaly run is flagged, the whole run
is considered detected.  Toggle with --adjustment / --no_adjustment.

Output CSV schema  (identical to test_all.py)
──────────────────────────────────────────────
encoder, detector, version, epoch, test_set, threshold_strategy,
accuracy, precision, recall, f1, auc,
recall_point, recall_contextual, recall_collective
"""

import argparse
import csv
import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────



LABEL_COL        = "anomaly"
LABEL_POINT_COL  = "Is_Anomaly_Point"
LABEL_CTX_COL    = "Is_Anomaly_Contextual"
LABEL_COLL_COL   = "Is_Anomaly_Collective"

ALL_LABEL_COLS = [LABEL_COL, LABEL_POINT_COL, LABEL_CTX_COL, LABEL_COLL_COL]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_directory(data_dir, require_labels=False, feature_cols=None):
    """
    Load all *.csv files in data_dir and concatenate them.

    Returns
    -------
    X : np.ndarray, shape [N, num_features]   – feature matrix
    y : np.ndarray, shape [N]      – overall anomaly label (int), or None
    y_typed : dict[str, np.ndarray] – per-type labels, or None
    ticker_ids : np.ndarray        – integer ticker index per row
    feature_cols : list[str]       – the feature columns used
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    if feature_cols is None:
        df_sample = pd.read_csv(csv_files[0])
        feature_cols = [
            col for col in df_sample.columns 
            if col not in ['Date', 'ticker', 'anomaly'] and not col.startswith('Is_Anomaly')
        ]

    X_parts, y_parts, y_point, y_ctx, y_coll = [], [], [], [], []
    ticker_ids = []

    for ticker_idx, fpath in enumerate(csv_files):
        df = pd.read_csv(fpath)

        # ── verify feature columns ─────────────────────────────────────────
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"  [SKIP] {os.path.basename(fpath)}: missing features {missing}")
            continue

        X_parts.append(df[feature_cols].values.astype(np.float32))
        ticker_ids.extend([ticker_idx] * len(df))

        # ── labels ────────────────────────────────────────────────────────
        has_labels = all(c in df.columns for c in ALL_LABEL_COLS)
        if require_labels and not has_labels:
            print(f"  [SKIP] {os.path.basename(fpath)}: label columns not found")
            X_parts.pop()
            ticker_ids = ticker_ids[: -len(df)]
            continue

        if has_labels:
            y_parts.append(df[LABEL_COL].astype(int).values)
            y_point.append(df[LABEL_POINT_COL].astype(int).values)
            y_ctx.append(df[LABEL_CTX_COL].astype(int).values)
            y_coll.append(df[LABEL_COLL_COL].astype(int).values)

    if not X_parts:
        raise ValueError(f"No valid CSV files could be loaded from: {data_dir}")

    X = np.concatenate(X_parts, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ticker_ids = np.array(ticker_ids, dtype=np.int32)

    if y_parts:
        y       = np.concatenate(y_parts)
        y_typed = {
            "point":       np.concatenate(y_point),
            "contextual":  np.concatenate(y_ctx),
            "collective":  np.concatenate(y_coll),
        }
    else:
        y, y_typed = None, None

    n_tickers = len(csv_files)
    print(f"  Loaded {X.shape[0]:,} rows from {n_tickers} tickers  ({data_dir})")
    if y is not None:
        rate = y.mean() * 100
        print(f"  Anomaly rate: {y.sum()} / {len(y)} = {rate:.2f}%")

    return X, y, y_typed, ticker_ids, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Threshold strategies
# ─────────────────────────────────────────────────────────────────────────────

def threshold_percentile(scores, percentile=98):
    return float(np.percentile(scores, percentile))


def threshold_pot(train_scores, test_scores, level=0.98, q=1e-5):
    """
    Peaks-Over-Threshold using model.spot.SPOT.
    Fallback logic handled by while loop shrinking lms.
    """
    from model.spot import SPOT
    lm = 0.99
    lms = level
    while True:
        try:
            s = SPOT(q)
            s.fit(train_scores, test_scores)
            s.initialize(level=lms, verbose=False)
        except Exception:
            lms *= 0.999
        else:
            break
            
    ret = s.run(dynamic=False)
    # If SPOT somehow still fails to return thresholds (e.g., empty), fallback to percentile
    if not ret.get('thresholds'):
        return float(np.percentile(train_scores, 99))
        
    return float(np.mean(ret['thresholds']) * lm)


# ─────────────────────────────────────────────────────────────────────────────
# Point-adjust (anomaly-state segment adjustment)
# ─────────────────────────────────────────────────────────────────────────────

def apply_point_adjust(y_pred, y_true):
    """
    If any timestep inside a labelled anomaly segment is predicted as
    anomalous, mark the entire segment as detected.
    """
    y_pred = y_pred.copy()
    i = 0
    while i < len(y_true):
        if y_true[i] == 1:
            j = i
            while j < len(y_true) and y_true[j] == 1:
                j += 1
            if np.any(y_pred[i:j] == 1):
                y_pred[i:j] = 1
            i = j
        else:
            i += 1
    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, scores):
    """
    Compute classification + AUC metrics.
    Handles edge cases (all-same label, empty positives, etc.)
    """
    out = {}
    try:
        out["accuracy"]  = accuracy_score(y_true, y_pred)
        out["precision"] = precision_score(y_true, y_pred, zero_division=0)
        out["recall"]    = recall_score(y_true, y_pred, zero_division=0)
        out["f1"]        = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        for k in ("accuracy", "precision", "recall", "f1"):
            out[k] = float("nan")

    try:
        if len(np.unique(y_true)) < 2:
            out["auc"] = "N/A"
        else:
            out["auc"] = roc_auc_score(y_true, scores)
    except Exception:
        out["auc"] = "N/A"

    return out


def compute_typed_recall(y_pred, y_typed):
    """
    Recall computed separately for each anomaly type using the
    pre-labelled type columns (no run-length inference needed).

    For each type, recall = fraction of that type's positive labels
    that were flagged as anomalous by the model.
    """
    result = {}
    for key, y_type in y_typed.items():
        positives = y_type == 1
        if positives.sum() == 0:
            result[f"recall_{key}"] = float("nan")
        else:
            result[f"recall_{key}"] = float(y_pred[positives].mean())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.4f}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate an Isolation Forest baseline "
                    "on the engineered stock anomaly dataset."
    )
    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--train_dir", type=str,
        default="./data/engineered_train_data_stat",
        help="Directory of per-ticker CSVs used to FIT the model.",
    )
    parser.add_argument(
        "--test_dirs", type=str, nargs="+",
        default=[
            "./data/engineered_test_synthetic_stat/Easy",
            "./data/engineered_test_synthetic_stat/Medium",
            "./data/engineered_test_synthetic_stat/Hard",
            "./data/engineered_test_data_stat/Easy",
            "./data/engineered_test_data_stat/Medium",
            "./data/engineered_test_data_stat/Hard",
        ],
        help="Directories of labelled per-ticker CSVs for evaluation.",
    )
    parser.add_argument(
        "--out_csv", type=str, default="test_results.csv",
        help="CSV file to APPEND results to.",
    )
    # ── model hyper-parameters ─────────────────────────────────────────────
    parser.add_argument(
        "--n_estimators", type=int, default=200,
        help="Number of trees in the Isolation Forest.",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected anomaly fraction (sklearn internal; overridden by our strategies).",
    )
    parser.add_argument(
        "--max_samples", type=str, default="auto",
        help="Subsample size per tree: 'auto', int, or float.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed.",
    )
    # ── threshold strategies ───────────────────────────────────────────────
    parser.add_argument(
        "--percentile", type=int, default=95,
        help="Percentile used by the 'percentile' threshold strategy.",
    )
    parser.add_argument(
        "--pot_q", type=float, default=0.90,
        help="Initialisation quantile q for POT / GPD fitting.",
    )
    parser.add_argument(
        "--pot_risk", type=float, default=1e-4,
        help="Target tail risk for POT threshold.",
    )
    # ── evaluation settings ────────────────────────────────────────────────
    parser.add_argument(
        "--adjustment", action="store_true", default=True,
        help="Apply anomaly-state segment adjustment (default: on).",
    )
    parser.add_argument(
        "--no_adjustment", dest="adjustment", action="store_false",
        help="Disable anomaly-state segment adjustment.",
    )
    parser.add_argument(
        "--version", type=str, default="default",
        help="Version tag written to the results CSV.",
    )

    args = parser.parse_args()

    # ── resolve max_samples ────────────────────────────────────────────────
    if args.max_samples == "auto":
        max_samples = "auto"
    else:
        try:
            max_samples = int(args.max_samples)
        except ValueError:
            max_samples = float(args.max_samples)

    # ══════════════════════════════════════════════════════════════════════
    # 1.  Load training data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 1 — Loading training data")
    print(f"{'='*60}")
    X_train, _, _, _, feature_cols = load_directory(args.train_dir, require_labels=False)
    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    # ══════════════════════════════════════════════════════════════════════
    # 2.  Fit Isolation Forest
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 2 — Fitting Isolation Forest")
    print(f"{'='*60}")
    print(f"  n_estimators : {args.n_estimators}")
    print(f"  contamination: {args.contamination}")
    print(f"  max_samples  : {max_samples}")
    print(f"  random_seed  : {args.random_seed}")

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=max_samples,
        random_state=args.random_seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    print("  Model fitted.")

    # ── training anomaly scores (higher = more anomalous) ─────────────────
    train_scores = -model.score_samples(X_train)

    # ── calibrate thresholds on train scores ──────────────────────────────
    thresholds = {
        "percentile": threshold_percentile(train_scores, percentile=args.percentile),
    }
    print(f"  Threshold [percentile]: {thresholds['percentile']:.6f}")
    print(f"  Threshold [       pot]: (calculated dynamically per test set)")

    # ══════════════════════════════════════════════════════════════════════
    # 3.  Evaluate on each test directory
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STEP 3 — Evaluating on test sets")
    print(f"{'='*60}")

    # Open CSV in append mode (creates header if file is new)
    fieldnames = [
        "encoder", "detector", "version", "epoch", "test_set",
        "threshold_strategy",
        "accuracy", "precision", "recall", "f1", "auc",
        "recall_point", "recall_contextual", "recall_collective",
    ]
    file_exists = os.path.exists(args.out_csv)
    csv_fh = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for test_dir in args.test_dirs:
        if not os.path.exists(test_dir):
            print(f"\n[SKIP] Directory not found: {test_dir}")
            continue

        # Derive test-set name (mirrors test_all.py convention)
        leaf          = os.path.basename(os.path.normpath(test_dir))
        parent        = os.path.basename(os.path.dirname(os.path.normpath(test_dir)))
        prefix        = "Synthetic" if "synthetic" in parent.lower() else "Real"
        test_set_name = f"{prefix}_{leaf}"

        print(f"\n{'─'*60}")
        print(f"Test set : {test_set_name}")
        print(f"Directory: {test_dir}")

        try:
            X_test, y_test, y_typed, _, _ = load_directory(
                test_dir, require_labels=True, feature_cols=feature_cols
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if y_test is None:
            print("  No labels found — skipping.")
            continue

        # Anomaly scores for test set
        test_scores = -model.score_samples(X_test)

        for strat in ("pot", "percentile"):
            if strat == "pot":
                thr = threshold_pot(train_scores, test_scores, level=args.pot_q, q=args.pot_risk)
            else:
                thr = thresholds[strat]

            y_pred = (test_scores >= thr).astype(int)

            if args.adjustment:
                y_pred = apply_point_adjust(y_pred, y_test)

            metrics      = compute_metrics(y_test, y_pred, test_scores)
            typed_recall = compute_typed_recall(y_pred, y_typed)

            auc_str = (
                metrics["auc"] if isinstance(metrics["auc"], str)
                else f"{metrics['auc']:.4f}"
            )
            print(
                f"  [{strat:>10s}]  "
                f"Acc={fmt(metrics.get('accuracy'))}  "
                f"P={fmt(metrics.get('precision'))}  "
                f"R={fmt(metrics.get('recall'))}  "
                f"F1={fmt(metrics.get('f1'))}  "
                f"AUC={auc_str}  "
                f"R_point={fmt(typed_recall.get('recall_point'))}  "
                f"R_ctx={fmt(typed_recall.get('recall_contextual'))}  "
                f"R_coll={fmt(typed_recall.get('recall_collective'))}"
            )

            row = {
                "encoder":            "None",
                "detector":           "IsolationForest",
                "version":            args.version,
                "epoch":              "N/A",
                "test_set":           test_set_name,
                "threshold_strategy": strat,
                "accuracy":           fmt(metrics.get("accuracy")),
                "precision":          fmt(metrics.get("precision")),
                "recall":             fmt(metrics.get("recall")),
                "f1":                 fmt(metrics.get("f1")),
                "auc":                fmt(metrics.get("auc")),
                "recall_point":       fmt(typed_recall.get("recall_point")),
                "recall_contextual":  fmt(typed_recall.get("recall_contextual")),
                "recall_collective":  fmt(typed_recall.get("recall_collective")),
            }
            writer.writerow(row)
            csv_fh.flush()

    csv_fh.close()
    print(f"\n{'='*60}")
    print(f"Results appended to: {args.out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
