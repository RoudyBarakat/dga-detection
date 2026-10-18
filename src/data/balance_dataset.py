"""Balanced version of the merged dataset.

Usage (from the repository root; arguments are optional):
    python src/data/balance_dataset.py [merged_csv_gz out_csv_gz [min_family seed]]

Keeps every legitimate domain. DGA families with fewer than `min_family` domains
(default 300) are merged into one family, `other_dga`. Every DGA family is then capped
at the same size, chosen so that DGA and legitimate domains are about equal in number.
Output columns: domain, label, family.
"""
import sys
from pathlib import Path

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

    out = pd.concat([legit, dga], ignore_index=True).sample(frac=1.0, random_state=seed)
    out[["domain", "label", "family"]].to_csv(out_path, index=False, compression="gzip")

    fam = dga["family"].value_counts()
    print(f"families < {min_family} merged into {OTHER}: {len(small)} "
          f"({int(sizes[small].sum())} domains)")
    print(f"cap per family: {cap}, families at cap: {(fam == cap).sum()} of {len(fam)}")
    print(f"DGA rows: {len(dga)} of {int(sizes.sum())}, legit rows: {len(legit)}, "
          f"total: {len(out)}")
    print(f"family size: min {fam.min()}, max {fam.max()}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or DEFAULTS))
