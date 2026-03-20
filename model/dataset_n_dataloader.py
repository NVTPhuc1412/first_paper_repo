import glob, os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    def __init__(self, config, data_tensor, labels_tensor = None, split='train', overlap=True):
        self.seq_len = config.seq_len
        if overlap != True and config.stride < self.seq_len:
            self.stride = self.seq_len
        else:
            self.stride = config.stride

        n_tickers, n_rows, _ = data_tensor.shape
        train_end = int(n_rows * config.train_size)

        self.split = split

        if split == 'train':
            self.data = data_tensor[:, :train_end, :]
        elif split == 'val':
            self.data = data_tensor[:, train_end:, :]
        elif split == 'test':
            if config.detector == 'Anomaly Transformer':
                test_start = n_rows % self.seq_len
                self.test_len = n_rows - test_start
            elif config.detector == 'TranAD':
                test_start = 0
                self.test_len = n_rows - self.seq_len + 1
            else:
                test_start = 0
                self.test_len = n_rows

            self.data = data_tensor[:, test_start:, :]
            self.labels = labels_tensor[:, -self.test_len:]
        else:
            self.data = data_tensor
            self.labels = labels_tensor

        split_len = self.data.shape[1]

        assert split_len >= self.seq_len, \
            f"Error: {split} split length ({split_len}) is smaller than seq_len ({self.seq_len})"

        self.n_per_ticker = (split_len - self.seq_len) // self.stride + 1
        self.n_samples = self.n_per_ticker * n_tickers

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ticker_idx = idx // self.n_per_ticker
        start_idx = (idx % self.n_per_ticker) * self.stride
        end_idx = start_idx + self.seq_len

        seq = self.data[ticker_idx, start_idx:end_idx, :]

        return seq

    def get_labels(self):
        return self.labels

    def get_type_labels(self):
        """Return per-type anomaly labels [n_tickers, T, 3] (Point, Contextual, Collective)."""
        return getattr(self, 'type_labels', None)


def get_data_tensor(data_root, is_test=False):
    """
    Returns:
        data_tensor: [n_tickers, n_timesteps, n_value_features]
        labels_tensor: [n_tickers, n_timesteps] (binary anomaly, test only)
        type_labels_tensor: [n_tickers, n_timesteps, 3] (per-type, test only)
        ticker_names: list of str
    """
    file_list = glob.glob(data_root + '/*.csv')
    n_tickers = len(file_list)
    ticker_names = [os.path.basename(path).replace('.csv', '') for path in file_list]

    # Read first file to check structure
    df_sample = pd.read_csv(file_list[0])

    # Dynamically select value columns (exclude Date, ticker, and any label columns)
    value_cols = [
        col for col in df_sample.columns 
        if col not in ['Date', 'ticker', 'anomaly'] and not col.startswith('Is_Anomaly')
    ]
    label_col = 'anomaly'
    type_label_cols = ['Is_Anomaly_Point', 'Is_Anomaly_Contextual', 'Is_Anomaly_Collective']

    n_value_feats = len(value_cols)
    n_rows = len(df_sample)

    data_tensor = torch.empty(n_tickers, n_rows, n_value_feats, dtype=torch.float32)

    if is_test:
        labels_tensor = torch.empty(n_tickers, n_rows, dtype=torch.float32)
        # Check if per-type columns exist in the data
        has_type_labels = all(col in df_sample.columns for col in type_label_cols)
        if has_type_labels:
            type_labels_tensor = torch.empty(n_tickers, n_rows, 3, dtype=torch.float32)
        else:
            type_labels_tensor = None

        for i, path in enumerate(file_list):
            df = pd.read_csv(path)
            data_tensor[i] = torch.from_numpy(df[value_cols].to_numpy().astype(np.float32))
            labels_tensor[i] = torch.from_numpy(df[label_col].to_numpy().astype(np.float32))
            if has_type_labels:
                type_labels_tensor[i] = torch.from_numpy(
                    df[type_label_cols].to_numpy().astype(np.float32)
                )
        return data_tensor, labels_tensor, type_labels_tensor, ticker_names
    else:
        for i, path in enumerate(file_list):
            df = pd.read_csv(path)
            data_tensor[i] = torch.from_numpy(df[value_cols].to_numpy().astype(np.float32))
        return data_tensor, None, None, ticker_names


def get_data_loaders(data_root, config, is_test=False):
    data_tensor, labels_tensor, type_labels_tensor, ticker_names = get_data_tensor(data_root, is_test)

    test_overlap = False if config.detector == 'Anomaly Transformer' else True

    if not is_test:
        train_ds = StockDataset(
            config, data_tensor,
            split='train'
        )
        val_ds = StockDataset(
            config, data_tensor,
            split='val'
        )

        train_dl = DataLoader(
            train_ds, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=False
        )

        return train_dl, val_dl, ticker_names
    else:
        test_ds = StockDataset(
            config, data_tensor,
            labels_tensor=labels_tensor,
            split='test', overlap=test_overlap
        )
        # Attach per-type labels if available
        if type_labels_tensor is not None:
            # Slice type_labels to match the test_len used for binary labels
            test_ds.type_labels = type_labels_tensor[:, -test_ds.test_len:]


        test_dl = DataLoader(
            test_ds, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=False
        )

        return test_dl, ticker_names


def get_data_loaders_whole(data_root, config):
    data_tensor, _, _, ticker_names = get_data_tensor(data_root, is_test=False)

    ds = StockDataset(config, data_tensor)
    dl = DataLoader(
        ds, batch_size=config.batch_size,
        num_workers=config.num_workers, shuffle=False
    )

    return dl, ticker_names