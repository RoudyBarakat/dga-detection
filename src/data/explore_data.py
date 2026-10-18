"""Exploratory analysis of the balanced dataset for the proposal's Data section.

Usage (from the repository root; arguments are optional):
    python src/data/explore_data.py [balanced_csv_gz out_dir]

Writes stats.json and data_overview.pdf/.png into <out_dir>.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = ["legit", "dga"]
NAMES = {"legit": "legitimate", "dga": "DGA"}
COLORS = {"legit": "#4C72B0", "dga": "#DD8452"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [ROOT / "data/processed/domains_balanced.csv.gz", ROOT / "proposal/figures"]


def entropy(s):
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def main(balanced_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(balanced_path, dtype=str, keep_default_na=False)
    df["sld"] = df["domain"].str.rsplit(".", n=1).str[0]
    df["tld"] = df["domain"].str.rsplit(".", n=1).str[-1]
    df["length"] = df["domain"].str.len()
    df["entropy"] = df["sld"].map(entropy)
    df["digit_ratio"] = df["sld"].str.count(r"[0-9]") / df["sld"].str.len()

    fam = df.loc[df["label"] == "dga", "family"].value_counts()
    tlds_per_family = df[df["label"] == "dga"].groupby("family")["tld"].nunique()
    stats = {
        "rows": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "families": int(fam.size),
        "family_size": {"min": int(fam.min()), "median": float(fam.median()),
                        "max": int(fam.max()), "at_cap": int((fam == fam.max()).sum())},
        "tlds_per_dga_family": {"median": float(tlds_per_family.median()),
                                "single_tld_families": int((tlds_per_family == 1).sum())},
        "other_dga": int(fam.get("other_dga", 0)),
    }
    for lab in LABELS:
        g = df[df["label"] == lab]
        top = g["tld"].value_counts(normalize=True)
        stats[lab] = {
            "domains": int(len(g)),
            "length_mean": round(float(g["length"].mean()), 1),
            "length_min_max": [int(g["length"].min()), int(g["length"].max())],
            "entropy_mean": round(float(g["entropy"].mean()), 2),
            "digit_ratio_mean": round(float(g["digit_ratio"].mean()), 3),
            "n_tlds": int(g["tld"].nunique()),
            "top_tld": [top.index[0], round(float(top.iloc[0]), 2)],
        }

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))

    axes[0].bar([NAMES[l] for l in LABELS], [stats[l]["domains"] for l in LABELS],
                color=[COLORS[l] for l in LABELS])
    axes[0].set_ylabel("domains")
    axes[0].set_title("(a) Class balance")

    axes[1].fill_between(range(fam.size), fam.values, step="mid", color=COLORS["dga"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("DGA family (sorted by size)")
    axes[1].set_ylabel("domains (log scale)")
    axes[1].set_title(f"(b) Domains per family ({fam.size} families)")

    for lab in LABELS:
        axes[2].hist(df.loc[df["label"] == lab, "length"].clip(upper=50), bins=range(3, 52),
                     density=True, alpha=0.6, color=COLORS[lab], label=NAMES[lab])
    axes[2].set_xlabel("domain length (characters)")
    axes[2].set_ylabel("density")
    axes[2].set_title("(c) Domain length")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "data_overview.pdf")
    fig.savefig(out / "data_overview.png", dpi=150)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main(*(sys.argv[1:3] or DEFAULTS))
