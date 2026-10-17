"""Family-balanced sample of the merged dataset.

Usage (from the repository root; arguments are optional):
    python src/data/balance_dataset.py [merged_csv_gz out_csv_gz [min_family cap seed]]

Keeps DGA families with at least `min_family` domains (default 300), samples at most
`cap` domains per family (default 3000), and samples as many legitimate domains as
DGA domains kept. Output columns: domain, label, family.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "data/processed/domains_merged.csv.gz",
            ROOT / "data/processed/domains_balanced.csv.gz"]


def main(merged_path, out_path, min_family=300, cap=3000, seed=42):
    min_family, cap, seed = int(min_family), int(cap), int(seed)
    df = pd.read_csv(merged_path, dtype=str, keep_default_na=False)

    dga = df[df["label"] == "dga"]
    sizes = dga["family"].value_counts()
    kept = sizes.index[sizes >= min_family]
    dropped = sizes[sizes < min_family]
    dga = dga[dga["family"].isin(kept)]
    dga = (dga.groupby("family", group_keys=False)
              .sample(frac=1.0, random_state=seed)
              .groupby("family").head(cap))

    legit = df[df["label"] == "legit"].sample(n=len(dga), random_state=seed)
    out = pd.concat([legit, dga]).sample(frac=1.0, random_state=seed)
    out.to_csv(out_path, index=False, compression="gzip")

    fam = dga["family"].value_counts()
    print(f"families kept: {len(kept)}, dropped (< {min_family}): {len(dropped)} "
          f"({int(dropped.sum())} domains)")
    print(f"DGA rows: {len(dga)}, legit rows: {len(legit)}, total: {len(out)}")
    print(f"family size: min {fam.min()}, max {fam.max()}, families at cap: {(fam == cap).sum()}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or DEFAULTS))
