import torch
import torch.nn as nn
from .utils import my_kl_loss


class TranADLoss(nn.Module):
    """
    TranAD Loss Module.
    Loss = Loss = (1/n) * MSE(tgt, x1) + (1 - 1/n) * MSE(tgt, x2)
    where n = current_epoch + 1
    """

    def __init__(self):
        super(TranADLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, tgt, x1_recon, x2_recon, epoch=4):
        """
        Compute TranAD loss with epoch-dependent weighting.

        Args:
            tgt: Target (last time step) [B, 1, d_model]
            x1_recon: Phase 1 reconstruction [B, 1, d_model]
            x2_recon: Phase 2 reconstruction [B, 1, d_model]
            epoch: Current epoch number (0-indexed)

        Returns:
            loss: Total weighted loss (scalar)
        """
        n = epoch + 1
        loss1 = self.mse(x1_recon, tgt)
        loss2 = self.mse(x2_recon, tgt)
        total_loss = (1.0 / n) * loss1 + (1.0 - 1.0 / n) * loss2
        total_loss = total_loss.mean()
        return total_loss


class AnomalyTransformerMinimaxLoss(nn.Module):
    """
    Anomaly Transformer Minimax Loss Module.

    1. Compute loss1 = rec_loss - k * series_loss (minimize)
    2. Compute loss2 = rec_loss + k * prior_loss (minimize)
    3. Backprop loss1 with retain_graph=True
    4. Backprop loss2
    """

    def __init__(self, config):
        super(AnomalyTransformerMinimaxLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_assoc = config.lambda_assoc
        self.win_size = config.seq_len

    def forward(self, x_input, x_recon, series_list, prior_list):
        """
        Compute minimax loss.

        Args:
            x_input: Original input [B, L, d_model] (before embedding in AT)
            x_recon: Reconstructed output [B, L, d_model]
            series_list: List of series-association from each layer [B, H, L, L]
            prior_list: List of prior-association from each layer [B, H, L, L]

        Returns:
            loss1: rec_loss - k * series_loss (for series-association gradient)
            loss2: rec_loss + k * prior_loss (for prior-association gradient)
        """
        series_loss = 0.0
        prior_loss = 0.0

        for u in range(len(prior_list)):
            # Normalize prior to sum to 1 (make it a proper probability distribution)
            prior_normalized = prior_list[u] / torch.unsqueeze(
                torch.sum(prior_list[u], dim=-1), dim=-1
            ).repeat(1, 1, 1, self.win_size)

            # Series loss: KL(series || prior) + KL(prior || series)
            # Detach prior to prevent gradient flow
            series_loss += (
                    torch.mean(my_kl_loss(series_list[u], prior_normalized.detach())) +
                    torch.mean(my_kl_loss(prior_normalized.detach(), series_list[u]))
            )

            # Prior loss: KL(prior || series) + KL(series || prior)
            # Detach series to prevent gradient flow
            prior_loss += (
                    torch.mean(my_kl_loss(prior_normalized, series_list[u].detach())) +
                    torch.mean(my_kl_loss(series_list[u].detach(), prior_normalized))
            )

        series_loss = series_loss / len(prior_list)
        prior_loss = prior_loss / len(prior_list)

        rec_loss = self.mse(x_recon, x_input)
        loss1 = rec_loss - self.lambda_assoc * series_loss  # Maximize series discrepancy
        loss2 = rec_loss + self.lambda_assoc * prior_loss  # Minimize prior discrepancy

        return loss1, loss2, rec_loss