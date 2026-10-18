"""Load the balanced dataset, split it and compute sample weights.

    from src.data.dataset import load_splits
    train, val, test = load_splits()

Each split is a DataFrame with columns domain, label, family, weight. The split is
80/10/10 train/validation/test, stratified by family, with a fixed seed so that every
model sees the same rows. Weights: legitimate and DGA weigh half each; within DGA a
domain's weight is proportional to 1/sqrt(family size). Weights have mean 1 on the
training set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BALANCED = ROOT / "data/processed/domains_balanced.csv.gz"
SPLIT_SEED = 42


def load(path=BALANCED):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def split(df, seed=SPLIT_SEED):
    """80/10/10 split within each family."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pos = df.groupby("family").cumcount() / df.groupby("family")["family"].transform("size")
    return df[pos < 0.8], df[(pos >= 0.8) & (pos < 0.9)], df[pos >= 0.9]


def sample_weights(df):
    """Weights computed from the family sizes in `df` (use on the training set)."""
    is_dga = (df["label"] == "dga").to_numpy()
    fam_size = df.groupby("family")["family"].transform("size").to_numpy()
    raw = np.where(is_dga, 1 / np.sqrt(fam_size), 1.0)
    half = len(df) / 2
    return np.where(is_dga, raw / raw[is_dga].sum() * half, half / (~is_dga).sum())


def load_splits(path=BALANCED, seed=SPLIT_SEED):
    train, val, test = split(load(path), seed)
    train = train.assign(weight=sample_weights(train))
    return train, val, test
