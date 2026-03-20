import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_proba=0.1, max_seq_len=500):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        pe_table = self.get_pe_table()
        self.register_buffer('pe_table', pe_table)

        self.dropout = nn.Dropout(dropout_proba)

    def get_pe_table(self):
        position_idxs = torch.arange(self.max_seq_len).unsqueeze(1)
        embedding_idxs = torch.arange(self.d_model).unsqueeze(0)

        angle_rads = position_idxs * 1 / torch.pow(10000, (2 * (embedding_idxs // 2)) / self.d_model)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pe_table = angle_rads.unsqueeze(0)  # So we can apply it to a batch

        return pe_table

    def forward(self, embeddings_batch):
        seq_len = embeddings_batch.size(1)
        pe_batch = self.pe_table[:, :seq_len].clone().detach()
        return self.dropout(embeddings_batch + pe_batch)


class TranAD(nn.Module):
    """
        TranAD: Transformer-based Anomaly Detection.

        Args:
        config: Configuration object with:
            - d_model: Feature dimension (will be used as feats)
            - effective_seq_len: Sequence length (n_window)
            - dropout: Dropout probability
            - tranad_nhead: Number of attention heads (auto-computed if None)
            - tranad_dim_feedforward: FFN dimension (auto-computed if None)

    Input/Output:
        - Input: src [B, L, feats], tgt [B, 1, feats] (last time step only)
        - Output: x1, x2 both [B, 1, feats] (reconstructions of last time step)
    """
    def __init__(self, config, skip_embedding=True):
        super(TranAD, self).__init__()
        # Extract parameters from config
        feats = config.tranad_feats
        n_window = config.seq_len
        dropout = config.dropout
        nheads = config.tranad_nheads
        dim_feedforward = config.tranad_dff
        dim_out = config.enc_in


        self.n_feats = feats

        # Build architecture
        if not skip_embedding:
            self.pos_encoder = PositionalEncoding(2 * feats, dropout, n_window)
        else:
            self.pos_encoder = nn.Identity()

        # Encoder
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=2 * feats,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            1
        )

        # Decoder 1 (Phase 1: without anomaly context)
        self.transformer_decoder1 = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=2 * feats,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            1
        )

        # Decoder 2 (Phase 2: with anomaly context)
        self.transformer_decoder2 = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=2 * feats,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            1
        )

        self.fcn = nn.Sequential(
            nn.Linear(2 * feats, feats),
            nn.Sigmoid()
        )
        self.final_proj = nn.Linear(feats, dim_out)


    def encode(self, src, c, tgt):
        """
        Encode source with context.

        Args:
            src: [B, L, feats] - Source sequence
            c: [B, L, feats] - Context (zeros in phase 1, residuals in phase 2)
            tgt: [B, L, feats] - Target sequence

        Returns:
            tgt: [B, L, 2*feats] - Repeated target
            memory: [B, L, 2*feats] - Encoder output
        """
        # Concatenate source with context
        src = torch.cat((src, c), dim=2)  # [B, L, 2*feats]

        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        tgt = tgt.repeat(1, 1, 2)  # [B, L, 2*feats]

        return tgt, memory

    def forward(self, src, tgt):
        """
        Forward pass through both phases.

        Args:
            src: [B, L, feats] - Full source sequence
            tgt: [B, 1, feats] - Last time step only (to be reconstructed)

        Returns:
            x1: [B, 1, feats] - Phase 1 reconstruction of last time step
            x2: [B, 1, feats] - Phase 2 reconstruction of last time step
        """
        # Phase 1 - Without anomaly context (c = 0)
        c = torch.zeros_like(src)
        tgt_encoded1, memory1 = self.encode(src, c, tgt)
        x1 = self.fcn(
            self.transformer_decoder1(tgt_encoded1, memory1)
        )

        # Phase 2 - With anomaly context (c = residual from phase 1)
        c = (x1 - src) ** 2
        tgt_encoded2, memory2 = self.encode(src, c, tgt)
        x2 = self.fcn(
            self.transformer_decoder2(tgt_encoded2, memory2)
        )

        x1 = self.final_proj(x1)
        x2 = self.final_proj(x2)

        return x1, x2