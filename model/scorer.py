import numpy as np
import torch
import torch.nn as nn

from .utils import my_kl_loss


class AnomalyScorer:
    """
    Computes per-timestep anomaly scores for a batch of windows and aggregates
    them back to the original sequence length.

    Scoring logic is detector-specific:
      - Anomaly Transformer: softmax-weighted association discrepancy * reconstruction error
        Uses non-overlapping windows so aggregation is a simple concatenation.
        The softmax is intentionally window-relative (see paper Section 3.3) —
        non-overlapping windows are required for this to be semantically valid.
      - TranAD: per-feature MSE between reconstructions and target (eq. 13).

    Args:
        config: Config object
    """

    def __init__(self, config):
        self.detector = config.detector
        self.seq_len = config.seq_len
        self.win_size = config.seq_len
        self.temperature = 50  # from original AT repo

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def score_batch(self, input, outputs):
        """
        Compute anomaly scores for a batch of windows.

        Args:
            input: [B, seq_len, enc_in]
            outputs: tuple returned by Model.forward()
                - AT: (x, x_recon, series, prior, sigmas)
                - TranAD: (x, x1, x2)

        Returns:
            scores: np.ndarray [B, seq_len]
        """
        if self.detector == 'Anomaly Transformer':
            return self._score_batch_at(input, outputs)
        elif self.detector == 'TranAD':
            return self._score_batch_tranad(input, outputs)
        else:
            raise NotImplementedError(f"Scorer not implemented for detector: {self.detector}")

    def aggregate_to_sequence(self, window_scores, seq_len):
        """
        Aggregate per-window scores to a sequence of length seq_len.

        AT:     window_scores [n_windows, seq_len]   → seq_scores [seq_len]
                Non-overlapping windows, simple concatenation + mean-pad tail.

        TranAD: window_scores [n_windows, 1, enc_in] → seq_scores [seq_len, enc_in]
                stride=1 so each window contributes exactly one timestep (the last).
                Squeeze the middle dim and stack directly, then mean-pad tail.

        Args:
            window_scores: np.ndarray — shape depends on detector (see above)
            seq_len: int — full sequence length for this split/ticker

        Returns:
            seq_scores: np.ndarray
                AT:     [seq_len]
                TranAD: [seq_len, enc_in]
        """
        if self.detector == 'TranAD':
            # [n_windows, 1, enc_in] -> [n_windows, enc_in]
            seq_scores = window_scores.squeeze(1)
            if len(seq_scores) < seq_len:
                pad_len = seq_len - len(seq_scores)
                pad_val = seq_scores.mean(axis=0, keepdims=True)  # [1, enc_in]
                pad = np.repeat(pad_val, pad_len, axis=0)
                seq_scores = np.concatenate([seq_scores, pad], axis=0)
            return seq_scores[-seq_len:]  # [seq_len, enc_in]

        else:
            # AT: [n_windows, seq_len] -> [seq_len]
            seq_scores = window_scores.reshape(-1)
            if len(seq_scores) < seq_len:
                pad_len = seq_len - len(seq_scores)
                pad_val = float(seq_scores.mean()) if len(seq_scores) > 0 else 0.0
                seq_scores = np.concatenate(
                    [seq_scores, np.full(pad_len, pad_val, dtype=np.float32)]
                )
            return seq_scores[-seq_len:]  # [seq_len]

    # -------------------------------------------------------------------------
    # Anomaly Transformer scoring
    # -------------------------------------------------------------------------

    def _score_batch_at(self, input, outputs):
        """
        Score formula (paper Section 3.3 / original repo):

            rec_loss = mean(MSE(x_encoded, x_recon), dim=-1)       [B, L]
            assoc    = sum_layers( KL(S||P) + KL(P||S) ) * temp    [B, L]
            metric   = softmax(-assoc, dim=-1)                      [B, L]
            score    = metric * rec_loss                            [B, L]

        The softmax is window-relative by design: it down-weights timesteps
        with high association discrepancy (normal) and up-weights those with
        low discrepancy (anomalous), then scales by reconstruction error.
        This is only semantically valid within a self-contained window —
        hence non-overlapping evaluation windows are required.
        """
        x_recon, series_list, prior_list, _ = outputs

        criterion = nn.MSELoss(reduction='none')
        rec_loss = torch.mean(criterion(input, x_recon), dim=-1)  # [B, L]

        win_size = input.shape[1]
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

        assoc = series_loss + prior_loss              # [B, L]
        metric = torch.softmax(-assoc, dim=-1)        # [B, L]
        score = metric * rec_loss                     # [B, L]

        return score.detach().cpu().numpy()

    # -------------------------------------------------------------------------
    # TranAD scoring
    # -------------------------------------------------------------------------

    def _score_batch_tranad(self, input, outputs):
        """
        Score formula (Algorithm 2, eq. 13 from TranAD paper):

            s = ½||O1 - Ŵ||₂ + ½||O2 - Ŵ||₂     per feature, [B, 1, enc_in]

        Both phase 1 (x1) and phase 2 (x2) reconstructions contribute equally.
        Scores are kept per-feature — thresholding happens feature-wise via POT,
        and a timestep is flagged anomalous if ANY feature exceeds its threshold
        (eq. 14: y = ∨ᵢ yᵢ).

        With stride=1 during test, each timestep is the last step of exactly
        one window, so window scores stack directly into a sequence.

        Args:
            outputs: (x, x1, x2)
                x:  [B, L, enc_in] - normalized input
                x1: [B, 1, enc_in] - phase 1 reconstruction of last timestep
                x2: [B, 1, enc_in] - phase 2 reconstruction of last timestep

        Returns:
            scores: np.ndarray [B, 1, enc_in] - per-feature score per window
        """
        x1, x2 = outputs
        tgt = input[:, -1:, :]  # [B, 1, enc_in]

        criterion = nn.MSELoss(reduction='none')
        score = 0.5 * criterion(x1, tgt) + 0.5 * criterion(x2, tgt)  # [B, 1, enc_in]

        return score.detach().cpu().numpy()