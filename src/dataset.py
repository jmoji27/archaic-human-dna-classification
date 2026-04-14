import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# One-hot encoding map — defined once at module level, not rebuilt per call
_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],  
}


class DNADataset(Dataset):
    """
    Dataset for genomic sequence classification.

    Parameters
    ----------
    data : str | pd.DataFrame
        Path to a CSV file or an already-loaded DataFrame.
        CSV must have columns: 'sequence', 'label'.
    L : int
        Fixed output sequence length (default 60).
    train : bool
        If True  → random crop per call (augmentation).
        If False → deterministic left crop (reproducible val/test).
    """

    def __init__(self, data: str | pd.DataFrame, L: int = 60, train: bool = True):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        # Drop sequences containing N (unresolved bases)
        #before = len(df)
        #df = df[~df["sequence"].str.contains("N", na=False)].reset_index(drop=True)
        #dropped = before - len(df)
        #if dropped:
            #print(f"[DNADataset] Dropped {dropped} sequences containing 'N'")

        self.df    = df
        self.L     = L
        self.train = train

    # internal helpers 
    #random cropping the one we will use

    def pad_sequence(self, seq, l =60):
        """Pad sequence to exactly l bases."""
        if len(seq) >= l:
            return seq
        return seq + "N" * (l - len(seq))

    def rand_crop(self, seq: str) -> str:
        """Crop or pad sequence to exactly self.L bases."""
        if len(seq) < self.L:
            # Pad
            return self.pad_sequence(seq, self.L)
        if len(seq) == self.L:
            return seq
        # Train → random crop  |  Val/Test → left crop (deterministic)
        start = random.randint(0, len(seq) - self.L) if self.train else 0
        return seq[start : start + self.L]
    
    # cropping but not ranodm used to check if model can memorize
    def _crop(self, seq: str) -> str:
        """Deterministic left crop or pad to self.L bases."""
        if len(seq) < self.L:
            return self.pad_sequence(seq, self.L)
        return seq[:self.L]
        
    
    

    @staticmethod
    def _one_hot(sequence: str) -> np.ndarray:
        """Convert a fixed-length DNA string to a (L, 4) float32 array."""
        return np.array([_MAPPING[b] for b in sequence], dtype=np.float32)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return the reverse complement of a DNA sequence."""
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(comp[b] for b in reversed(seq))

    # dataset interface

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = self.rand_crop(row["sequence"])

        # Reverse complement augmentation — train only, 50% chance
        # Biologically valid: both strands carry the same information
        if self.train and random.random() < 0.5:
            seq = self._reverse_complement(seq)

        x = torch.from_numpy(self._one_hot(seq))          # (L, 4)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y