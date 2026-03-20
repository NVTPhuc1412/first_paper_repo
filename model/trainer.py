import os
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .loss_functions import TranADLoss, AnomalyTransformerMinimaxLoss


class EarlyStopping:
    def __init__(self, patience=7, delta=0, epoch_dir=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.epoch_dir = epoch_dir

    def __call__(self, signal, model, epoch):
        # Save every epoch to the versioned folder
        if self.epoch_dir is not None:
            epoch_path = os.path.join(self.epoch_dir, f'epoch_{epoch + 1:03d}.pth')
            torch.save(model.state_dict(), epoch_path)
            print(f'Saved checkpoint → {epoch_path}')

        # Early stopping logic
        score = -signal
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, version=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        self.encoder = config.encoder
        self.detector = config.detector
        self.num_epochs = config.num_epochs

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        fmt = "%Y-%m-%d_%H-%M"
        run_name = f"{config.encoder}_{config.detector}_{datetime.now().strftime(fmt)}"
        self.writer = SummaryWriter(log_dir=f'runs/{run_name}')

        # Loss Functions
        if self.detector == 'TranAD':
            self.loss_f = TranADLoss()
        elif self.detector == 'Anomaly Transformer':
            self.loss_f = AnomalyTransformerMinimaxLoss(config)

        # Checkpoint directory
        os.makedirs('./checkpoints', exist_ok=True)
        version_str = version if version is not None else 'default'
        epoch_dir = f'./checkpoints/{model.enc_type}-{model.dec_type}_{version_str}_epochs'
        os.makedirs(epoch_dir, exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience=config.patience,
            delta=config.min_delta,
            epoch_dir=epoch_dir,
        )
        self.global_step = 0

    def train(self):
        print(f"Starting training for {self.encoder} -> {self.detector}...")
        train_hist = []
        val_hist = []

        for epoch in range(self.num_epochs):
            start_time = time.time()

            self.model.train()
            train_losses = []

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )

            for batch in progress_bar:
                # batch shape: [B, L, C]
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                if self.detector == 'TranAD':
                    x1_rec, x2_rec = outputs
                    loss = self.loss_f(batch[:,-1:,:], x1_rec, x2_rec, epoch)
                    loss.backward()

                elif self.detector == 'Anomaly Transformer':
                    x_recon, series, prior, _ = outputs

                    loss1, loss2, rec_loss = self.loss_f(
                        batch, x_recon, series, prior
                    )
                    loss = loss1
                    self.writer.add_scalar('Loss/l1', loss1.item(), self.global_step)
                    self.writer.add_scalar('Loss/l2', loss2.item(), self.global_step)

                    loss1.backward(retain_graph=True)
                    loss2.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item() * batch.size(0))
                progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                self.global_step += 1

            train_loss_avg = sum(train_losses) / len(self.train_loader.dataset)
            val_loss = self.validate(epoch)

            self.writer.add_scalar('Loss/train_loss', train_loss_avg, epoch)

            if isinstance(val_loss, tuple):
                self.writer.add_scalar('Loss/val1_epoch', val_loss[0], epoch)
                self.writer.add_scalar('Loss/val2_epoch', val_loss[1], epoch)
                self.writer.add_scalar('Loss/valrec_epoch', val_loss[2], epoch)
            else:
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)

            epoch_time = time.time() - start_time
            val_display = val_loss[0] if isinstance(val_loss, tuple) else val_loss
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss_avg:.6f} | Val Loss (rec): {val_display:.6f} | Time: {epoch_time:.1f}s")

            train_hist.append(train_loss_avg)
            val_hist.append(val_loss[0] if isinstance(val_loss, tuple) else val_loss)

            if isinstance(val_loss, tuple):
                signal = val_loss[2]
            else:
                signal = val_loss

            self.early_stopping(signal, self.model, epoch=epoch)

            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.writer.close()
        return train_hist, val_hist

    def validate(self, epoch):
        """Validation loop."""
        self.model.eval()
        val_losses = 0
        if self.detector == 'Anomaly Transformer':
            val_losses2 = 0
            rec_losses = 0
        else:
            val_losses2 = rec_losses = None

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                outputs = self.model(batch)

                if self.detector == 'TranAD':
                    x1_rec, x2_rec = outputs

                    loss = self.loss_f(batch[:,-1:,:], x1_rec, x2_rec, epoch)
                    val_losses += loss.item() * batch.size(0)

                elif self.detector == 'Anomaly Transformer':
                    x_recon, series, prior, _ = outputs

                    loss1, loss2, rec_loss = self.loss_f(
                        batch, x_recon, series, prior
                    )
                    val_losses += loss1.item() * batch.size(0)
                    val_losses2 += loss2.item() * batch.size(0)
                    rec_losses += rec_loss.item() * batch.size(0)

        if val_losses2 is None:
            return val_losses/len(self.val_loader.dataset)
        else:
            return (
                val_losses/len(self.val_loader.dataset),
                val_losses2/len(self.val_loader.dataset),
                rec_losses/len(self.val_loader.dataset)
            )
