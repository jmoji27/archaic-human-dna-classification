from itertools import product
from collections import Counter
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# One-hot encoding map.
# N → [0,0,0,0] (zero vector) instead of [0.25,0.25,0.25,0.25].
# The 0.25 encoding was a uniform distribution that the model could detect
# in N-runs and use to infer the original read length (= class signal).
# A zero vector is still distinguishable from real bases but carries no
# spurious positional/length information.
_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],  # N is all zeros, not 0.25 each, to avoid length-based class signal
}


def variable_length_collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate sequences of different lengths into a padded batch.

    Padding is done to the longest sequence IN THIS BATCH, not to a global
    MAX_LEN. Because batches are shuffled randomly, the padding length changes
    every epoch and is uncorrelated with class — the model cannot learn
    'zeros start at position X → class Y'.

    Padded positions are zero-filled (matching the N encoding above),
    so the model sees a consistent zero signal for all non-data positions.

    This replaces the fixed-L padding in DNADataset and works correctly
    with any model that uses global pooling before the classifier head
    (e.g. CNN1D with AdaptiveMaxPool1d, which your architecture already has).
    """
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    # Shape: (batch, max_len_in_batch, 4) — zero-padded
    padded = torch.zeros(len(xs), max_len, 4, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    return padded, torch.stack(ys)


class DNADataset(Dataset):
    """
    Dataset for genomic sequence classification.

    Parameters
    ----------
    data : str | pd.DataFrame
        Path to a CSV file or an already-loaded DataFrame.
        CSV must have columns: 'sequence', 'label'.
    train : bool
        If True  -> reverse-complement augmentation applied at 50%.
        If False -> no augmentation (reproducible val/test).

    Note: sequences are returned at their ORIGINAL length with no padding.
    Use `variable_length_collate` as the DataLoader's collate_fn so that
    batches are padded dynamically to the longest sequence in each batch.
    This eliminates fixed-length padding as a class signal.
    """

    def __init__(self, data: str | pd.DataFrame, train: bool = True):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle dataset

        self.df     = df
        self.train  = train
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)

    @staticmethod
    def _one_hot(sequence: str) -> np.ndarray:
        """Convert a DNA string to a (L, 4) float32 array. L = actual sequence length."""
        return np.array([_MAPPING[b] for b in sequence], dtype=np.float32)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return the reverse complement of a DNA sequence."""
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(comp[b] for b in reversed(seq))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = row["sequence"].upper()

        # Reverse complement augmentation — train only, 50% chance.
        # Biologically valid: both strands carry the same information.
        if self.train and random.random() < 0.5:
            seq = self._reverse_complement(seq)

        # No cropping or padding here — sequences are returned at their
        # actual length. variable_length_collate handles batch-level padding.
        x = torch.from_numpy(self._one_hot(seq))           # (L, 4)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Legacy classes kept for reference — not used in the main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DNADatasetNoPadding(Dataset):
    """Original no-padding dataset — superseded by DNADataset above."""

    def __init__(self, data: str | pd.DataFrame, train: bool = True):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        self.df    = df
        self.train = train

    @staticmethod
    def _one_hot(sequence: str) -> np.ndarray:
        mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0,0,0,0]}
        return np.array([mapping[b] for b in sequence], dtype=np.float32)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(comp[b] for b in reversed(seq))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = row["sequence"].upper()
        if self.train and random.random() < 0.5:
            seq = self._reverse_complement(seq)
        x = torch.from_numpy(self._one_hot(seq))
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


class DnaDatasetKmer(Dataset):
    """
    K-mer tokenization dataset.
    Not used in the main pipeline but kept for experimentation.
    """

    def __init__(
            self, data: str | pd.DataFrame,
            k: int = 3, L: int = 60, stride: int = 1,
            train: bool = True, normalize: bool = True,
    ):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        self.df        = df
        self.k         = k
        self.L         = L
        self.stride    = stride
        self.train     = train
        self.normalize = normalize

        self.kmer_vocab  = self._build_kmer_vocab()
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.kmer_vocab)}
        self.vocab_size  = len(self.kmer_vocab)
        print(f"[K-mer Dataset] k={k}, stride={stride}, vocab_size={self.vocab_size}")

    def _build_kmer_vocab(self) -> list:
        bases = ['A', 'C', 'G', 'T']
        return [''.join(p) for p in product(bases, repeat=self.k)]

    def _extract_kmers(self, seq: str) -> np.ndarray:
        kmers = []
        for i in range(0, len(seq) - self.k + 1, self.stride):
            kmer = seq[i:i+self.k]
            if 'N' not in kmer:
                kmers.append(kmer)
        kmer_counts = Counter(kmers)
        features = np.zeros(self.vocab_size, dtype=np.float32)
        for kmer, count in kmer_counts.items():
            if kmer in self.kmer_to_idx:
                features[self.kmer_to_idx[kmer]] = count
        if self.normalize and features.sum() > 0:
            features = features / features.sum()
        return features

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(comp[b] for b in reversed(seq))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = row["sequence"].upper()
        if self.train and random.random() < 0.5:
            seq = self._reverse_complement(seq)
        x = torch.from_numpy(self._extract_kmers(seq))
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y

        

        
        