"""Step 2: Markov chain score, learned from legitimate domains.

A Markov chain answers: "given the previous letter(s), how likely is the next
one?" We learn these probabilities from LEGITIMATE training names, then score
each domain by how expected each of its letters was.

  order 1: look at the previous 1 character  (after 'q', 'u' is very likely)
  order 2: look at the previous 2 characters (after 'th', 'e' is very likely)

Example (order 1): 'google' -> g->o, o->o, o->g, g->l, l->e, e->END: all common
                   -> high score. 'xkqjzp' -> x->k, k->q, q->j ...: all rare
                   -> low score.

The score is the AVERAGE log10 probability per step, so long and short names
can be compared (closer to 0 = more normal, more negative = more random).

Same leakage rule as the n-gram features: counts come only from the training
set, and training rows are scored with counts from the OTHER folds.
"""
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def transitions(s: str, order: int) -> list[str]:
    """All (context + next character) pieces of s.

    '^' pads the start (so the first letters also have a context) and '$'
    marks the end. Each piece is order+1 characters: the first `order` are the
    context, the last one is the next character.
      transitions('ab', 1) -> ['^a', 'ab', 'b$']
      transitions('ab', 2) -> ['^^a', '^ab', 'ab$']
    """
    s = "^" * order + s + "$"
    return [s[i:i + order + 1] for i in range(len(s) - order)]


def count_transitions(texts, order: int) -> Counter:
    """Count every (context + next character) piece over many strings."""
    counts = Counter()
    for s in texts:
        counts.update(transitions(s, order))
    return counts


class MarkovModel:
    """Turns transition counts into probabilities and per-domain scores."""

    def __init__(self, counts: Counter, n_symbols: int):
        self.counts = counts
        # How often each context appears (with any next character).
        self.context_counts = defaultdict(int)
        for piece, c in counts.items():
            self.context_counts[piece[:-1]] += c
        # Number of possible next characters (letters, digits, '-', end marker).
        self.n_symbols = n_symbols

    def log_prob(self, piece: str) -> float:
        """log10 P(next character | context), with +1 smoothing.

        Smoothing: a transition never seen still gets a small probability,
        so the log is defined.
        """
        context = piece[:-1]
        return math.log10((self.counts.get(piece, 0) + 1)
                          / (self.context_counts.get(context, 0) + self.n_symbols))

    def score(self, s: str, order: int) -> float:
        """Average log10 probability per step of string s."""
        pieces = transitions(s, order)
        return sum(self.log_prob(p) for p in pieces) / len(pieces)


def markov_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                    orders=(1, 2), n_folds: int = 5, seed: int = 0,
                    text_col: str = "text", label_col: str = "label"):
    """Compute Markov scores for train, validation and test.

    One column per order: markov1, markov2 (average log10 probability per step;
    higher = more like a legitimate name).

    Counts come from the LEGIT rows of the training set only:
      - validation / test rows: counts from all legit training rows;
      - training rows: counts from the legit training rows of the OTHER folds.
    Same folds as the n-gram features (same seed), so both use the same split.

    Returns three DataFrames (train, val, test), with the same index as the inputs.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(train))
    is_legit = (train[label_col] == "legit").to_numpy()

    out_train, out_val, out_test = {}, {}, {}
    for order in orders:
        total = count_transitions(train.loc[is_legit, text_col], order)
        fold_counts = [count_transitions(train.loc[is_legit & (fold == k), text_col], order)
                       for k in range(n_folds)]
        # Number of different next characters seen in legit names (incl. '$').
        n_symbols = len({piece[-1] for piece in total})

        col = f"markov{order}"
        parts = []
        for k in range(n_folds):
            model_k = MarkovModel(total - fold_counts[k], n_symbols)
            texts_k = train.loc[fold == k, text_col]
            parts.append(pd.Series([model_k.score(s, order) for s in texts_k],
                                   index=texts_k.index))
        out_train[col] = pd.concat(parts).loc[train.index]

        model_all = MarkovModel(total, n_symbols)
        out_val[col] = pd.Series([model_all.score(s, order) for s in val[text_col]],
                                 index=val.index)
        out_test[col] = pd.Series([model_all.score(s, order) for s in test[text_col]],
                                  index=test.index)

    return pd.DataFrame(out_train), pd.DataFrame(out_val), pd.DataFrame(out_test)
