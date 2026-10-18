"""Balanced, split and weighted version of the merged dataset.

Usage (from the repository root; arguments are optional):
    python src/data/balance_dataset.py [merged_csv_gz out_csv_gz [min_family seed]]

Keeps every legitimate domain. DGA families with fewer than `min_family` domains
(default 300) are merged into one family, `other_dga`. Every DGA family is then capped
at the same size, chosen so that DGA and legitimate domains are about equal in number.
Each row gets a split (80/10/10 train/val/test, stratified by family) and a sample
weight: legitimate and DGA weigh half each, and within DGA a domain's weight is
proportional to 1/sqrt(family size). Weights have mean 1.
Output columns: domain, label, family, split, weight.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "data/processed/domains_merged.csv.gz",
            ROOT / "data/processed/domains_balanced.csv.gz"]
OTHER = "other_dga"


def matching_cap(sizes, target):
    """Smallest per-family cap whose capped total reaches `target`."""
    lo, hi = 1, int(sizes.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if sizes.clip(upper=mid).sum() < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def split_by_family(df, seed):
    """80/10/10 split within each family (rows are already shuffled)."""
    rank = df.groupby("family").cumcount()
    size = df.groupby("family")["family"].transform("size")
    pos = rank / size
    return np.select([pos < 0.8, pos < 0.9], ["train", "val"], "test")


def main(merged_path, out_path, min_family=300, seed=42):
    min_family, seed = int(min_family), int(seed)
    df = pd.read_csv(merged_path, dtype=str, keep_default_na=False)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    legit = df[df["label"] == "legit"]
    dga = df[df["label"] == "dga"].copy()
    sizes = dga["family"].value_counts()
    small = sizes.index[sizes < min_family]
    dga.loc[dga["family"].isin(small), "family"] = OTHER

    cap = matching_cap(dga["family"].value_counts(), len(legit))
    dga = dga.groupby("family").head(cap)

    out = pd.concat([legit, dga], ignore_index=True)
    out["split"] = split_by_family(out, seed)

    fam_size = out.groupby("family")["family"].transform("size")
    is_dga = out["label"] == "dga"
    raw = np.where(is_dga, 1 / np.sqrt(fam_size), 1.0)
    half = len(out) / 2
    out["weight"] = np.where(is_dga, raw / raw[is_dga].sum() * half,
                             half / (~is_dga).sum())
    out["weight"] = out["weight"].round(6)

    out = out.sample(frac=1.0, random_state=seed)
    out[["domain", "label", "family", "split", "weight"]].to_csv(
        out_path, index=False, compression="gzip")

    fam = dga["family"].value_counts()
    w = out.loc[is_dga, ["family", "weight"]].drop_duplicates("family")["weight"]
    print(f"families < {min_family} merged into {OTHER}: {len(small)} "
          f"({int(sizes[small].sum())} domains)")
    print(f"cap per family: {cap}, families at cap: {(fam == cap).sum()} of {len(fam)}")
    print(f"DGA rows: {len(dga)} of {int(sizes.sum())}, legit rows: {len(legit)}, "
          f"total: {len(out)}")
    print(f"family size: min {fam.min()}, max {fam.max()}; "
          f"DGA weight max/min: {w.max() / w.min():.1f}")
    print("split:", out["split"].value_counts().to_dict())


if __name__ == "__main__":
    main(*(sys.argv[1:] or DEFAULTS))
