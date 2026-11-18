"""Step 1a: basic character features, computed on the 'text' column.

'text' = subdomain + name without dots (see split.py), e.g.
'mail.google.com' -> 'mailgoogle'. The suffix (TLD) is NOT used here, so these
features stay valid for the without-TLD experiments.
"""
import math
from collections import Counter

import pandas as pd


def shannon_entropy(s: str) -> float:
    """How random a string looks, in bits per character.

    Low  for repetitive strings: 'aaaa'     -> 0.0
    High for varied strings:     'xk3qz9wp' -> 3.0 (8 different characters)
    """
    counts = Counter(s)            # how many times each character appears
    n = len(s)
    return abs(-sum(c / n * math.log2(c / n) for c in counts.values()))  # abs: avoid -0.0


def basic_features(text: pd.Series) -> pd.DataFrame:
    """Compute simple character features for every string in 'text'.

    Returns a table with one row per domain and these columns:
      length             number of characters
      digit_ratio        share of characters that are digits (0-9)
      hyphen_ratio       share of characters that are '-'
      unique_char_ratio  number of different characters / length
      entropy            Shannon entropy (see shannon_entropy)
      longest_digit_run  longest block of digits in a row ('ab1234c' -> 4)
      starts_with_digit  1 if the first character is a digit, else 0
    """
    length = text.str.len()
    safe_len = length.clip(lower=1)   # avoids dividing by 0 (no empty texts expected)

    out = pd.DataFrame(index=text.index)
    out["length"] = length
    out["digit_ratio"] = text.str.count(r"[0-9]") / safe_len
    out["hyphen_ratio"] = text.str.count("-") / safe_len
    out["unique_char_ratio"] = text.map(lambda s: len(set(s))) / safe_len
    out["entropy"] = text.map(shannon_entropy)
    # findall gives every block of digits, e.g. 'ab12c345' -> ['12', '345'];
    # we keep the length of the longest one (0 if there are no digits).
    out["longest_digit_run"] = text.str.findall(r"[0-9]+").map(
        lambda blocks: max((len(b) for b in blocks), default=0))
    out["starts_with_digit"] = text.str.match(r"^[0-9]").astype(int)
    return out
