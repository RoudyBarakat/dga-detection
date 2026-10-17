"""Merge the domain sources into one file with columns: domain, label, family.

Usage (from the repository root; arguments are optional):
    python src/data/build_dataset.py [mendeley_zip kaggle_zip tranco_zip out_csv_gz]

Sources:
  - DGA: Mendeley (Gregorio & Cansian 2025), DGA rows only: domain, family, type
  - DGA: Kaggle gtkcyber/dga-dataset, DGA rows only: isDGA, domain, host, subclass
    (the `domain` column is the SLD, truncated at '-' for some rows, so `host` is used)
  - Legitimate: Tranco top 1M (list 64VVX), no header: rank, domain

The legitimate rows of the Mendeley (Majestic) and Kaggle (Alexa) files are not used.
label is "legit" or "dga"; family is "legit" for legitimate domains, else the DGA family
using the Mendeley/DGArchive names.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "data/raw/mendeley_dga_2025.zip", ROOT / "data/raw/kaggle_dga_2021.zip",
            ROOT / "data/raw/tranco_64VVX_top1m.zip", ROOT / "data/processed/domains_merged.csv.gz"]

# Kaggle subclass -> Mendeley/DGArchive family. Matched by domain shape (length, charset,
# TLD set), since no DGA domain occurs in both datasets (different collection years).
KAGGLE_FAMILY = {
    "gameoverdga": "gameover",   # newGOZ DGA: 21-28 alnum chars, .com/.net/.org/.biz
    "newgoz": "gameover",        # identical pattern to gameoverdga and Mendeley gameover
    "goz": "gameover_p2p_a0b92382",  # original GOZ P2P DGA: 15-32 letters, no digits
    "bamital": "bamital",
    "cryptolocker": "cryptolocker",
    "necurs": "necurs",
    "nivdort": "nivdort",        # word pairs (.net); not in Mendeley, kept as its own family
}

VALID = r"^[a-z0-9-]+(\.[a-z0-9-]+)+$"


def normalize(s):
    return s.str.strip().str.lower().str.rstrip(".")


def main(mendeley_zip, kaggle_zip, tranco_zip, out_path):
    m = pd.read_csv(mendeley_zip, compression="zip", usecols=["domain", "family"],
                    dtype=str, keep_default_na=False)
    m = m[m["family"] != "legit"].assign(source="mendeley")

    k = pd.read_csv(kaggle_zip, compression="zip", usecols=["isDGA", "host", "subclass"],
                    dtype=str, keep_default_na=False)
    k = k[k["isDGA"] == "dga"]
    unknown = set(k["subclass"]) - set(KAGGLE_FAMILY)
    if unknown:
        raise ValueError(f"unmapped Kaggle subclasses: {unknown}")
    k = pd.DataFrame({"domain": k["host"], "family": k["subclass"].map(KAGGLE_FAMILY),
                      "source": "kaggle"})

    t = pd.read_csv(tranco_zip, compression="zip", header=None, names=["rank", "domain"],
                    dtype=str, keep_default_na=False)
    t = pd.DataFrame({"domain": t["domain"], "family": "legit", "source": "tranco"})

    df = pd.concat([m, k, t], ignore_index=True)
    df["domain"] = normalize(df["domain"])
    df["label"] = (df["family"] != "legit").map({True: "dga", False: "legit"})
    print("rows per source:", df["source"].value_counts().to_dict())

    invalid = ~df["domain"].str.match(VALID)
    print("invalid domain strings dropped:", int(invalid.sum()),
          df.loc[invalid].groupby("source").size().to_dict(),
          df.loc[invalid, "domain"].head(5).tolist())
    df = df[~invalid]

    # A domain is kept only if every source agrees on its label and family.
    g = df.groupby("domain").agg(n_label=("label", "nunique"), n_family=("family", "nunique"))
    label_conflict = g.index[g["n_label"] > 1]
    family_conflict = g.index[(g["n_label"] == 1) & (g["n_family"] > 1)]
    conflicts = df[df["domain"].isin(label_conflict) & (df["label"] == "dga")]
    print("domains in Tranco and in a DGA list (dropped):", len(label_conflict))
    print("  by DGA family:", conflicts["family"].value_counts().head(8).to_dict())
    print("  examples:", conflicts["domain"].head(8).tolist())
    print("domains with DGA family conflict dropped:", len(family_conflict))
    df = df[~df["domain"].isin(label_conflict.union(family_conflict))]

    before = len(df)
    df = df.drop_duplicates("domain")
    print("duplicate rows removed:", before - len(df))

    out = df[["domain", "label", "family"]].sort_values(["label", "family", "domain"])
    out.to_csv(out_path, index=False, compression="gzip")

    print("final rows:", len(out))
    print("label counts:", out["label"].value_counts().to_dict())
    print("DGA families:", out.loc[out["label"] == "dga", "family"].nunique())
    print("kept rows per source:", df["source"].value_counts().to_dict())


if __name__ == "__main__":
    main(*(sys.argv[1:5] or DEFAULTS))
