import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

MAX_LEN = 120  # center-crop anything longer than this


def danq_collate(batch):
    """
    Same as variable_length_collate but returns a padding mask too.
    Only used by DanQ — other models keep using variable_length_collate.
    """
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    padded = torch.zeros(len(xs), max_len, 4, dtype=torch.float32)
    mask = torch.ones(len(xs), max_len, dtype=torch.bool)  # True = padding
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = False
    return padded, torch.stack(ys), mask


class DanqDataset(Dataset):
    """
    Drop-in replacement for DNADataset, used only by DanQ.
    Adds:
      - center-crop to MAX_LEN
      - random base substitution augmentation
      - returns same (x, y) tuple — danq_collate adds the mask
    """

    def __init__(self, data: str | pd.DataFrame, train: bool = True):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        self.df     = df
        self.train  = train
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    @staticmethod
    def _one_hot(seq: str) -> np.ndarray:
        return np.array([_MAPPING[b] for b in seq], dtype=np.float32)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(comp[b] for b in reversed(seq))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"].upper()

        if self.train:
            if random.random() < 0.5:
                seq = self._reverse_complement(seq)
            if random.random() < 0.15:
                seq = list(seq)
                for i in range(len(seq)):
                    if random.random() < 0.02:
                        seq[i] = random.choice('ACGT')
                seq = ''.join(seq)

        # Center-crop to MAX_LEN
        if len(seq) > MAX_LEN:
            start = (len(seq) - MAX_LEN) // 2
            seq = seq[start:start + MAX_LEN]

        x = torch.from_numpy(self._one_hot(seq))
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y