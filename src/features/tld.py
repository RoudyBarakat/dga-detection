"""Step 4: TLD (suffix) features, computed on the 'suffix' column.

The suffix is a category ('com', 'net', 'co.uk', 'ru'...), not a number, so it
is turned into columns the model can use:

  - one on/off column per common suffix: tld_com, tld_net, ... (1 if the domain
    has that suffix, else 0). The list of common suffixes is learned from the
    TRAINING set only; every other suffix goes into tld_other.
  - suffix_parts   number of labels in the suffix ('com' -> 1, 'co.uk' -> 2)
  - suffix_len     number of characters in the suffix
  - tld_alt_dns    1 if the suffix belongs to an alternative DNS system
                   (.bazar, .bit, ... see EXTRA_SUFFIXES in split.py)

These are the features switched off in the "without TLD" experiments.
"""
import pandas as pd

from src.features.split import EXTRA_SUFFIXES


class TldEncoder:
    """Learns the most common suffixes on the training set, then encodes any set."""

    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.common = []   # filled by fit()

    def fit(self, suffix: pd.Series) -> "TldEncoder":
        """Remember the top_k most frequent suffixes (training set only)."""
        self.common = suffix.value_counts().index[: self.top_k].tolist()
        return self

    def transform(self, suffix: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=suffix.index)
        # One on/off column per common suffix ('.' replaced so names stay simple).
        for s in self.common:
            out[f"tld_{s.replace('.', '_')}"] = (suffix == s).astype(int)
        out["tld_other"] = (~suffix.isin(self.common)).astype(int)
        # A few numbers about the suffix itself.
        out["suffix_parts"] = suffix.str.count(r"\.") + 1
        out["suffix_len"] = suffix.str.len()
        out["tld_alt_dns"] = suffix.isin(EXTRA_SUFFIXES).astype(int)
        return out


def tld_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                 top_k: int = 50, suffix_col: str = "suffix"):
    """Encode the suffix of train, validation and test.

    The list of common suffixes comes from the training set only, so the
    validation and test sets get exactly the same columns.
    Returns three DataFrames (train, val, test), with the same index as the inputs.
    """
    enc = TldEncoder(top_k).fit(train[suffix_col])
    return (enc.transform(train[suffix_col]),
            enc.transform(val[suffix_col]),
            enc.transform(test[suffix_col]))
