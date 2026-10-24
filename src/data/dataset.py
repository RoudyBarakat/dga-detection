"""Load the saved train/validation/test splits and compute sample weights.

    from src.data.dataset import load_splits
    train, val, test = load_splits()

The splits are made once by notebooks/00_data_splits.ipynb (80/10/10, stratified by
family, seed 42) and saved in data/splits/. Every file is checked against
data/splits/manifest.json, so every model trains and is evaluated on exactly the same
rows. Each split has the columns domain, label, family; train also gets weight:
legitimate and DGA weigh half each, and within DGA a domain's weight is proportional
to 1/sqrt(family size). Weights have mean 1 on the training set.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits"


def load_split(name, splits_dir=SPLITS):
    """Read one split (train, val or test) after checking it against the manifest."""
    manifest = json.loads((splits_dir / "manifest.json").read_text())
    fname = f"{name}.csv.gz"
    info = manifest["files"][fname]
    path = splits_dir / fname
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError(f"{path} does not match manifest.json; "
                         "re-run notebooks/00_data_splits.ipynb")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(df) == info["rows"]
    return df


def sample_weights(df):
    """Weights computed from the family sizes in `df` (use on the training set)."""
    is_dga = (df["label"] == "dga").to_numpy()
    fam_size = df.groupby("family")["family"].transform("size").to_numpy()
    raw = np.where(is_dga, 1 / np.sqrt(fam_size), 1.0)
    half = len(df) / 2
    return np.where(is_dga, raw / raw[is_dga].sum() * half, half / (~is_dga).sum())


def load_splits(splits_dir=SPLITS):
    train, val, test = (load_split(name, splits_dir) for name in ("train", "val", "test"))
    train = train.assign(weight=sample_weights(train))
    return train, val, test
