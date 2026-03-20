import numpy as np
import torch
import os
import itertools
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from .scorer import AnomalyScorer
from .spot import SPOT


class Tester:
    """
    Handles end-to-end testing for anomaly detection models.

    Procedure:
      1. Accept regular overlapping train/val loaders for threshold calibration
      2. Build a non-overlapping loader for the test split only
         (stride = seq_len preserves AT softmax window-relative semantics)
      3. Fit POT threshold on combined train+val scores — no test leakage
      4. Collect test scores, aggregate per ticker, evaluate
      5. Apply optional anomaly-state segment adjustment
      6. Compute per-ticker metrics, macro-average, and per-type recall

    Args:
        model:        trained Model instance
        train_loader: DataLoader for train split (threshold calibration)
        val_loader:   DataLoader for val split (threshold calibration)
        test_loader:  DataLoader for test split (evaluation)
        config:       Config object
    """

    def __init__(self, model, train_loader, val_loader, test_loader, config, checkpoint_path=None):
        self.model = model
        self.config = config
        self.device = config.device
        self.checkpoint_path = checkpoint_path
        self.detector = config.detector
        self.threshold_strategy = getattr(config, 'threshold_strategy', 'pot')
        self.encoder = config.encoder
        self.seq_len = config.seq_len

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.n_tickers = test_loader.dataset.data.shape[0]
        self.n_per_ticker = test_loader.dataset.n_per_ticker

        self.scorer = AnomalyScorer(config)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def test(self, adjustment=True, each_ticker=False):
        """
        Run full test procedure.

        Args:
            adjustment: bool — apply anomaly-state segment adjustment (default True)
                        Follows original AT paper evaluation protocol.
                        Set to False for stricter point-wise evaluation.
            each_ticker: bool — whether to return per-ticker metrics (default False)

        Returns:
            results: dict with per-ticker metrics and 'macro' averaged entry.
                     Each entry contains: accuracy, precision, recall, f1, auc.
        """
        # Check for cached threshold
        threshold_path = None
        thresholds = None
        threshold_scalar = None
        
        if self.checkpoint_path:
            import os
            # Use test_loader dir name to differentiate cached thresholds across datasets (Easy/Medium/etc)
            try:
                test_dir_name = os.path.basename(os.path.normpath(self.test_loader.dataset.data_dir))
                cache_suffix = f'_{test_dir_name}_{self.threshold_strategy}_thresholds.npy'
            except Exception:
                cache_suffix = f'_{self.threshold_strategy}_thresholds.npy'
                
            threshold_path = self.checkpoint_path.replace('.pth', cache_suffix)
            if os.path.exists(threshold_path):
                print(f"Loading cached thresholds from {threshold_path}...")
                loaded_th = np.load(threshold_path)
                if self.detector == 'TranAD':
                    thresholds = loaded_th
                    print(f"Loaded POT thresholds per feature: {np.round(thresholds, 6)}")
                else:
                    threshold_scalar = float(loaded_th[0])
                    print(f"Loaded POT threshold (scalar): {threshold_scalar:.6f}")

        print("Collecting test scores...")
        test_window_scores = self._collect_scores_windows(self.test_loader)

        if thresholds is None and threshold_scalar is None:
            print("Collecting train + val scores for threshold calibration...")
            ratio = getattr(self.config, 'calibration_ratio', 1.0)
            
            train_limit = max(1, int(len(self.train_loader) * ratio)) if ratio < 1.0 else None
            val_limit = max(1, int(len(self.val_loader) * ratio)) if ratio < 1.0 else None

            train_scores = self._collect_scores_flat(self.train_loader, limit=train_limit)
            val_scores = self._collect_scores_flat(self.val_loader, limit=val_limit)
            calibration_scores = np.concatenate([train_scores, val_scores])

            # Threshold Calculation
            if self.threshold_strategy == 'percentile':
                q = getattr(self.config, 'threshold_percentile', 98)
                if self.detector == 'TranAD':
                    thresholds = np.percentile(calibration_scores, q, axis=0)
                    print(f"Percentile ({q}%) thresholds per feature: {np.round(thresholds, 6)}")
                    if threshold_path:
                        np.save(threshold_path, thresholds)
                        print(f"Saved computed thresholds to {threshold_path}")
                else:
                    threshold_scalar = float(np.percentile(calibration_scores, q))
                    print(f"Percentile ({q}%) threshold (scalar): {threshold_scalar:.6f}")
                    if threshold_path:
                        np.save(threshold_path, np.array([threshold_scalar], dtype=np.float64))
                        print(f"Saved computed thresholds to {threshold_path}")
            else:
                if self.detector == 'TranAD':
                    # TranAD: calibration_scores: [N_calib, enc_in], test flat: [N_test, enc_in]
                    thresholds = self._pot_threshold(calibration_scores, test_window_scores.squeeze(1))
                    print(f"POT thresholds per feature: {np.round(thresholds, 6)}")
                    if threshold_path:
                        np.save(threshold_path, thresholds)
                        print(f"Saved computed thresholds to {threshold_path}")
                else:
                    # Anomaly Transformer: scalar scores [N_calib] and [N_test]
                    threshold_scalar = self._pot_threshold(calibration_scores, test_window_scores.reshape(-1))
                    print(f"POT threshold (scalar): {threshold_scalar:.6f}")
                    if threshold_path:
                        np.save(threshold_path, np.array([threshold_scalar], dtype=np.float64))
                        print(f"Saved computed thresholds to {threshold_path}")

        if self.detector != 'TranAD' and threshold_scalar is not None:
            threshold = threshold_scalar

        # Aggregate per ticker: list of [T_test] arrays
        test_seq_len = self.test_loader.dataset.test_len
        ticker_scores = self._reshape_to_tickers(test_window_scores, test_seq_len)

        # Per-ticker evaluation
        labels = self.test_loader.dataset.get_labels().numpy()  # [n_tickers, T_test]
        results = {}
        all_preds, all_gts, all_scores = [], [], []

        for t in range(self.n_tickers):
            scores_t = ticker_scores[t]   # AT: [T_test] | TranAD: [T_test, enc_in]
            gt_t = labels[t].astype(int)

            if self.detector == 'TranAD':
                # Adapted from eq. 14 for financial data: average score across
                # features instead of per-feature OR-gate, since stock features
                # are correlated transformations of the same price series.
                mean_score = scores_t.mean(axis=-1)           # [T_test]
                mean_threshold = thresholds.mean()
                pred_t = (mean_score >= mean_threshold).astype(int)
                auc_scores_t = mean_score
            else:
                pred_t = (scores_t > threshold).astype(int)
                auc_scores_t = scores_t

            if adjustment:
                pred_t = self._anomaly_state_adjustment(pred_t, gt_t)

            all_preds.append(pred_t)
            all_gts.append(gt_t)
            all_scores.append(auc_scores_t)

            if each_ticker:
                acc = accuracy_score(gt_t, pred_t)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    gt_t, pred_t, average='binary', zero_division=0
                )
                try:
                    auc = roc_auc_score(gt_t, auc_scores_t)
                except ValueError:
                    auc = float('nan')

                results[f'ticker_{t}'] = {
                    'accuracy': acc, 'precision': prec,
                    'recall': rec, 'f1': f1, 'auc': auc
                }

                print(f"Ticker {t:>3} | Acc: {acc:.4f} | Prec: {prec:.4f} | "
                      f"Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # Macro average across all tickers
        macro_gt = np.concatenate(all_gts)
        macro_pred = np.concatenate(all_preds)
        macro_scores = np.concatenate(all_scores)

        macro_acc = accuracy_score(macro_gt, macro_pred)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            macro_gt, macro_pred, average='binary', zero_division=0
        )
        try:
            macro_auc = roc_auc_score(macro_gt, macro_scores)
        except ValueError:
            macro_auc = float('nan')

        results['macro'] = {
            'accuracy': macro_acc, 'precision': macro_prec,
            'recall': macro_rec, 'f1': macro_f1, 'auc': macro_auc
        }
        print(f"\nMacro  | Acc: {macro_acc:.4f} | Prec: {macro_prec:.4f} | "
              f"Rec: {macro_rec:.4f} | F1: {macro_f1:.4f} | AUC: {macro_auc:.4f}")

        # ── Per-type Recall ───────────────────────────────────────────────
        type_labels_tensor = self.test_loader.dataset.get_type_labels()
        if type_labels_tensor is not None:
            type_names = ['Point', 'Contextual', 'Collective']
            type_labels_np = type_labels_tensor.numpy()  # [n_tickers, T_test, 3]
            print("\nPer-Type Recall:")
            for col_idx, tname in enumerate(type_names):
                # Gather all predictions where this type's ground truth is 1
                type_gt_all = []
                type_pred_all = []
                for t in range(self.n_tickers):
                    type_gt_t = type_labels_np[t, :, col_idx].astype(int)
                    pred_t = all_preds[t]
                    # Only evaluate on timesteps where this type is present
                    mask = type_gt_t == 1
                    if mask.sum() > 0:
                        type_gt_all.append(type_gt_t[mask])
                        type_pred_all.append(pred_t[mask])
                if len(type_gt_all) > 0:
                    type_gt_concat = np.concatenate(type_gt_all)
                    type_pred_concat = np.concatenate(type_pred_all)
                    type_recall = type_pred_concat.sum() / len(type_pred_concat)
                    n_total = len(type_pred_concat)
                    n_detected = int(type_pred_concat.sum())
                    print(f"  {tname:12s} | Recall: {type_recall:.4f}  ({n_detected}/{n_total})")
                    results[f'recall_{tname.lower()}'] = type_recall
                else:
                    print(f"  {tname:12s} | No samples")
                    results[f'recall_{tname.lower()}'] = float('nan')

        return results

    # -------------------------------------------------------------------------
    # Score collection
    # -------------------------------------------------------------------------

    def _collect_scores_flat(self, loader, limit=None):
        """
        Score all windows and return scores collapsed to 2D for threshold fitting.

        AT:     [N_windows, seq_len]    → reshape(-1)        → [N_total]
        TranAD: [N_windows, 1, enc_in]  → squeeze + reshape  → [N_windows, enc_in]
        """
        window_scores = self._collect_scores_windows(loader, limit=limit)
        if self.detector == 'TranAD':
            return window_scores.squeeze(1)   # [N_windows, enc_in]
        return window_scores.reshape(-1)       # [N_total]

    def _collect_scores_windows(self, loader, limit=None):
        """
        Run all batches through the model and collect per-window scores.

        Returns:
            np.ndarray [total_windows, seq_len]
        """
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            if limit is not None:
                loader_iter = itertools.islice(loader, limit)
                total_iters = limit
            else:
                loader_iter = loader
                total_iters = len(loader)

            for batch in tqdm(loader_iter, total=total_iters, desc='Scoring', leave=False):
                batch = batch.to(self.device)
                outputs = self.model(batch)
                scores = self.scorer.score_batch(batch,outputs)  # [B, win_size]
                all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)

    def _pot_threshold(self, init_scores, test_scores,
                       q=1e-5, level=0.98, lm=0.99):
        """
        Fit SPOT instances on calibration scores, run on test scores, return thresholds.
        Handles per-feature 2D arrays for TranAD and 1D scalars for Anomaly Transformer.

        Args:
            init_scores: np.ndarray - train+val scores
            test_scores: np.ndarray - test scores
            q:     detection level (risk), default 1e-5
            level: init threshold quantile, default 0.98
            lm:    threshold multiplier, default 0.99

        Returns:
            thresholds: np.ndarray [enc_in] (TranAD) or float (Anomaly Transformer)
        """
        if self.detector == 'Anomaly Transformer':
            init_scores = init_scores.reshape(-1, 1)
            test_scores = test_scores.reshape(-1, 1)

        enc_in = init_scores.shape[1]
        thresholds = np.zeros(enc_in, dtype=np.float64)

        for i in range(enc_in):
            lms = level
            while True:
                try:
                    s = SPOT(q)
                    s.fit(init_scores[:, i], test_scores[:, i])
                    s.initialize(level=lms, verbose=False)
                except Exception:
                    lms *= 0.999
                else:
                    break
            ret = s.run(dynamic=False)
            thresholds[i] = float(np.mean(ret['thresholds']) * lm)

        if self.detector == 'Anomaly Transformer':
            return float(thresholds[0])
        return thresholds

    def _reshape_to_tickers(self, window_scores, seq_len):
        """
        Split [n_tickers * n_per_ticker, win_size] into per-ticker sequences.

        Returns list of np.ndarray, one per ticker, each of length seq_len.
        """
        ticker_seq_scores = []
        for t in range(self.n_tickers):
            start = t * self.n_per_ticker
            end = start + self.n_per_ticker
            windows_t = window_scores[start:end]
            seq_t = self.scorer.aggregate_to_sequence(windows_t, seq_len)
            ticker_seq_scores.append(seq_t)
        return ticker_seq_scores

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    @staticmethod
    def _anomaly_state_adjustment(pred, gt):
        """
        If any point within a contiguous ground-truth anomaly segment is
        detected, mark the entire segment as detected.

        This is the evaluation protocol from the original AT paper.
        See: https://github.com/thuml/Anomaly-Transformer/issues/14
        """
        pred = pred.copy()
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):  # backfill segment
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
                for j in range(i, len(gt)):  # forward-fill segment
                    if gt[j] == 0:
                        break
                    if pred[j] == 0:
                        pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred