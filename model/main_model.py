import torch
import torch.nn as nn

from .embedder.TimesNet_embedder import TimesNetEmbedder
from .embedder.PatchTST_embedder import PatchTSTEmbedder
from .embedder.iTransformer import iTransformerEmbedder
from .detector.TranAD import TranAD
from .detector.AnomalyTransformer import AnomalyTransformer


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.enc_type = config.encoder
        self.dec_type = config.detector

        # Initialize encoder
        if config.encoder == 'TimesNet':
            self.encoder = TimesNetEmbedder(config)
        elif config.encoder == 'PatchTST':
            self.encoder = PatchTSTEmbedder(config)
        elif config.encoder == 'iTransformer':
            self.encoder = iTransformerEmbedder(config)
        else:
            self.encoder = nn.Identity()

        if config.encoder is not None:
            dec_skip_embed = True
        else:
            dec_skip_embed = False

        # Initialize detector
        if config.detector == 'TranAD':
            self.detector = TranAD(config, skip_embedding=dec_skip_embed)

        elif config.detector == 'Anomaly Transformer':
            self.detector = AnomalyTransformer(config, skip_embedding=dec_skip_embed)

        else:
            raise ValueError(f"Unknown detector type: {config.detector}")

    def forward(self, x):
        """
        Forward pass through encoder and detector.

        Args:
            x: Input tensor [B, L, C] where C is enc_in
        Returns:
            Depends on detector type:
            - TranAD: (x_encoded, tgt, x1_recon, x2_recon)
                x_encoded: [B, L, d_model] - full encoded sequence
                tgt: [B, 1, d_model] - last time step (target)
                x1_recon, x2_recon: [B, 1, d_model] - reconstructions of last time step
            - Anomaly Transformer: (x_input, x_encoded, x_recon, series, prior, sigmas)
                x_encoded: [B, L, d_model] - encoded representation
                x_recon: [B, L, d_model] - reconstruction
                series, prior, sigmas: attention components
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x_encoded = self.encoder(x)  # [B, L, d_model]

        if self.dec_type == 'TranAD':
            if self.enc_type == 'TimesNet':
                tgt = x_encoded[:, -1:, :]
            else:
                tgt = x[:, -1:, :]
            x1, x2 = self.detector(x_encoded, tgt)


            x1 = self._denorm(x1, means, stdev)
            x2 = self._denorm(x2, means, stdev)

            return x1, x2

        elif self.dec_type == 'Anomaly Transformer':
            enc_out, series, prior, sigmas = self.detector(x_encoded)

            enc_out = self._denorm(enc_out, means, stdev)

            return enc_out, series, prior, sigmas

        else:
            raise ValueError(f"Unknown detector type: {self.dec_type}")

    def _denorm(self, x, mean, std):
        x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.size(1), 1))
        x = x + (mean[:, 0, :].unsqueeze(1).repeat(1, x.size(1), 1))
        return x



