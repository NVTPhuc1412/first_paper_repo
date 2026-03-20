"""
Configuration class for Anomaly Detection Pipeline.

Supports detectors: TranAD, Anomaly Transformer.
Supports encoders: None (raw features), TimesNet, PatchTST, iTransformer.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # ============= Data Parameters =============
    enc_in: int = 8  # Number of input features (e.g., OHLCV + indicators)
    seq_len: int = 96  # Input sequence length
    pred_len: int = 0  # Prediction length (0 for anomaly detection only)
    stride: int = 1
    train_size: float = 0.8

    # ============= Model Architecture =============
    encoder: Literal['TimesNet', 'PatchTST', 'iTransformer', None] = None  # Embedder type
    detector: Literal['TranAD', 'Anomaly Transformer'] = 'TranAD'  # Detector type
    activation: Literal['relu', 'gelu'] = 'gelu'

    # ============= TimesNet Specific Parameters =============
    tn_top_k: int = 3  # Number of top frequencies to consider
    tn_num_kernels: int = 3  # Number of inception kernels
    tn_d_model: int = 32
    tn_d_ff: int = 64
    tn_elayers: int = 2

    # ============= PatchTST Specific Parameters =============
    patch_len: int = 16
    patch_stride: int = 8
    patch_d_model: int = 64
    patch_d_ff: int = 256
    patch_nheads: int = 8
    patch_elayers: int = 3

    # ============= iTransformer Specific Parameters =============
    itran_d_model: int = 64
    itran_d_ff: int = 256
    itran_nheads: int = 8
    itran_e_layers: int = 3

    # ============= TranAD Specific Parameters =============
    tranad_feats: int = 64
    tranad_nheads: int = 8
    tranad_dff: int = 256

    # ============= Anomaly Transformer Specific Parameters =============
    at_d_in: int = 64
    at_nheads: int = 8
    at_elayers: int = 3
    at_d_model: int = 64
    at_d_ff: int = 256
    lambda_assoc: float = 0.0002

    # ============= Regularization =============
    dropout: float = 0.15  # Dropout rate

    # ============= Training Parameters =============
    batch_size: int = 256
    num_workers: int = 0
    learning_rate: float = 1e-4
    num_epochs: int = 15

    # ============= Early Stopping Parameters =============
    patience: int = 5
    min_delta: float = 1e-5

    # ============= Testing Parameters =============
    threshold_strategy: Literal['pot', 'percentile'] = 'pot'
    threshold_percentile: int = 98
    calibration_ratio: float = 0.1

    # ============= Device =============
    device: str = 'cuda'  # 'cuda' or 'cpu'

    def __post_init__(self):
        # Calculate effective sequence length for PatchTST
        if self.encoder in ['PatchTST', 'iTransformer']:
            self.n_patches = (self.seq_len - self.patch_len) // self.patch_stride + 2
            self.at_d_in = self.enc_in
            self.tranad_feats = self.enc_in
            self.tranad_dff = self.enc_in * 8
            self.tranad_nheads = 4 if self.enc_in == 8 else max(1, self.enc_in // 2)
        elif self.encoder == 'TimesNet':
            self.at_d_in  = self.tn_d_model
            self.at_d_ff = self.at_d_model * 4
            self.tranad_feats = self.tn_d_model
            self.tranad_dff = self.tranad_feats * 4
        else:
            self.at_d_in = self.enc_in
            self.tranad_feats = self.enc_in
            self.tranad_dff = self.enc_in * 8
            self.tranad_nheads = self.enc_in // 2

