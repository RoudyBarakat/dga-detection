"""Step 1b: n-gram frequency features, learned from legitimate domains.

An n-gram is a group of n characters in a row. For 'google':
  2-grams: ^g go oo og gl le e$
  3-grams: ^go goo oog ogl gle le$
'^' and '$' mark the start and end of the string, so the model also learns
how real names usually begin and end.

Idea: real names reuse common letter groups ('th', 'er', 'ing'); random DGA
names contain rare ones ('qx', 'zkj'). We count how often every n-gram appears
in LEGITIMATE training domains, then score each domain by how common its
n-grams are.

Leakage rule: the counts come only from the training set, never from
validation or test. Inside the training set we use 5 "folds": the features of
a training row are computed from counts that do NOT include that row's fold.
Otherwise every legit training domain would only contain n-grams that are
"already known", and the model would learn a pattern that does not hold on
new data (see ngram_features).
"""
import math
from collections import Counter

import numpy as np
import pandas as pd


def ngrams(s: str, n: int) -> list[str]:
    """All n-grams of s, with start/end markers: ngrams('ab', 2) -> ['^a', 'ab', 'b$']."""
    s = "^" + s + "$"
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def count_ngrams(texts, n: int) -> Counter:
    """Count every n-gram over a collection of strings."""
    counts = Counter()
    for s in texts:
        counts.update(ngrams(s, n))
    return counts


class NgramModel:
    """Turns n-gram counts into per-domain scores."""

    def __init__(self, counts: Counter, top_k: int = 500, rare_rate: float = 1e-6):
        self.counts = counts
        self.total = sum(counts.values())    # number of n-grams seen
        self.vocab = len(counts) + 1         # number of different n-grams (+1 for "unseen")
        # The top_k most common n-grams in legitimate domains.
        self.top = {g for g, _ in counts.most_common(top_k)}
        # An n-gram is "rare" if it makes up less than rare_rate of all n-grams
        # (1e-6 = fewer than 1 in a million, about 10 times in our legit data).
        # A rate instead of a fixed count, so the fold models (built on 80% of
        # the data) and the full model use the same definition.
        self.rare_below = rare_rate * self.total

    def log_freq(self, g: str) -> float:
        """log10 of how frequent n-gram g is in legit domains.

        +1 smoothing: an n-gram never seen still gets a small frequency
        instead of 0, so the log is defined.
        """
        return math.log10((self.counts.get(g, 0) + 1) / (self.total + self.vocab))

    def score(self, s: str, n: int) -> tuple[float, float, float, float]:
        """The 4 features of one string (see ngram_features)."""
        grams = ngrams(s, n)
        logs = [self.log_freq(g) for g in grams]
        return (
            sum(logs) / len(logs),                                    # mean
            min(logs),                                                # rarest
            sum(self.counts.get(g, 0) < self.rare_below for g in grams) / len(grams),  # rare
            sum(g in self.top for g in grams) / len(grams),           # top
        )


def _score_all(texts: pd.Series, model: NgramModel, n: int) -> pd.DataFrame:
    rows = [model.score(s, n) for s in texts]
    cols = [f"ng{n}_mean", f"ng{n}_min", f"ng{n}_rare", f"ng{n}_top"]
    return pd.DataFrame(rows, index=texts.index, columns=cols)


def ngram_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   ns=(2, 3), n_folds: int = 5, seed: int = 0,
                   text_col: str = "text", label_col: str = "label"):
    """Compute n-gram features for train, validation and test.

    For each n in ns (default 2-grams and 3-grams), 4 columns:
      ng{n}_mean    average log-frequency of the domain's n-grams (higher = more normal)
      ng{n}_min     log-frequency of its rarest n-gram
      ng{n}_rare    share of its n-grams that are rare in legit training domains
                    (less than 1 in a million; replaces "never seen", which was
                    almost always 0 because nearly every n-gram appears somewhere)
      ng{n}_top     share of its n-grams among the 500 most common legit n-grams

    Counts come from the LEGIT rows of the training set only:
      - validation / test rows: counts from all legit training rows;
      - training rows: counts from the legit training rows of the OTHER folds
        (out-of-fold), so a row never helps compute its own features.

    Returns three DataFrames (train, val, test), with the same index as the inputs.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(train))   # fold number of each training row
    is_legit = (train[label_col] == "legit").to_numpy()

    out_train, out_val, out_test = [], [], []
    for n in ns:
        # Counts over all legit training rows, and per fold.
        total = count_ngrams(train.loc[is_legit, text_col], n)
        fold_counts = [count_ngrams(train.loc[is_legit & (fold == k), text_col], n)
                       for k in range(n_folds)]

        # Training rows: score fold k with the counts of all OTHER folds.
        parts = []
        for k in range(n_folds):
            model_k = NgramModel(total - fold_counts[k])   # Counter subtraction
            parts.append(_score_all(train.loc[fold == k, text_col], model_k, n))
        out_train.append(pd.concat(parts).loc[train.index])

        # Validation and test rows: counts from all legit training rows.
        model_all = NgramModel(total)
        out_val.append(_score_all(val[text_col], model_all, n))
        out_test.append(_score_all(test[text_col], model_all, n))

    return (pd.concat(out_train, axis=1),
            pd.concat(out_val, axis=1),
            pd.concat(out_test, axis=1))
