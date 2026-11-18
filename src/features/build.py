"""Build all hand-crafted features (steps 0-4 of notebooks/03_features.ipynb) in one call.

    from src.features.build import build_features
    feats = build_features(train, val, test)
    feats["train"], feats["val"], feats["test"]   # one table per split (74 columns)
    feats["name_cols"], feats["tld_cols"]         # for the with/without-TLD runs

Computing everything takes 10-20 minutes, so the result is saved once in
data/processed/features/ and loaded from there next time (a few seconds).
If the split or the feature code changes, delete that folder (or pass
use_cache=False) to recompute.
"""
from pathlib import Path

import pandas as pd

from src.features.handcrafted import basic_features
from src.features.markov import markov_features
from src.features.ngrams import ngram_features
from src.features.split import add_split_columns
from src.features.tld import tld_features
from src.features.vowels import vowel_features

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/features"


def _compute(train, val, test):
    # Step 0: subdomain / name / suffix / text
    train, val, test = (add_split_columns(d) for d in (train, val, test))

    # Step 1a: basic character features (no training)
    basic = [basic_features(d["text"]) for d in (train, val, test)]
    # Step 1b: n-gram features (learned from legit training names, out-of-fold)
    ngram = list(ngram_features(train, val, test))
    ngram = [d.drop(columns="ng2_rare") for d in ngram]   # always 0, see notebook 03
    # Step 2: Markov scores (learned from legit training names, out-of-fold)
    markov = list(markov_features(train, val, test))
    # Step 3: vowel / consonant features (no training)
    vowel = [vowel_features(d["text"]) for d in (train, val, test)]
    # Step 4: TLD features (common suffixes learned from the training set)
    tld = list(tld_features(train, val, test))

    name_cols = list(basic[0].columns) + list(ngram[0].columns) \
        + list(markov[0].columns) + list(vowel[0].columns)
    tld_cols = list(tld[0].columns)
    tables = [pd.concat([b, n, m, v, t], axis=1)
              for b, n, m, v, t in zip(basic, ngram, markov, vowel, tld)]
    return tables, name_cols, tld_cols


def build_features(train, val, test, use_cache=True, cache_dir=CACHE):
    """All 74 features for train, val and test (same row order as the inputs)."""
    files = {name: Path(cache_dir) / f"{name}.pkl" for name in ("train", "val", "test")}
    if use_cache and all(f.exists() for f in files.values()):
        tables = [pd.read_pickle(files[name]) for name in ("train", "val", "test")]
        cols = pd.read_pickle(Path(cache_dir) / "columns.pkl")
        name_cols, tld_cols = cols["name_cols"], cols["tld_cols"]
        # Safety check: the cached rows must be the rows we were given.
        for t, d in zip(tables, (train, val, test)):
            if len(t) != len(d) or not t.index.equals(d.index):
                raise ValueError("cached features do not match the split; "
                                 "delete data/processed/features/ and rerun")
    else:
        tables, name_cols, tld_cols = _compute(train, val, test)
        if use_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            for name, t in zip(("train", "val", "test"), tables):
                t.to_pickle(files[name])
            pd.to_pickle({"name_cols": name_cols, "tld_cols": tld_cols},
                         Path(cache_dir) / "columns.pkl")
    return {"train": tables[0], "val": tables[1], "test": tables[2],
            "name_cols": name_cols, "tld_cols": tld_cols}
