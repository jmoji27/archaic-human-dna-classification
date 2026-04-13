import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, data, L=40, train=True):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data

        
        self.L = L
        self.train = train

    def crop(self, seq):
        if len(seq) <= self.L:
            return seq

        if self.train:
            start = random.randint(0, len(seq) - self.L)
        else:
            start = 0

        return seq[start:start+self.L]

    def one_hot(self, sequence):
        mapping = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1]
        }

        encoded = [mapping.get(b, [0,0,0,0]) for b in sequence]
        return np.array(encoded, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.crop(row["sequence"])
        x = self.one_hot(seq)
        y = row["label"]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y