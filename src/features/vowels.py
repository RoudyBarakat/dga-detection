"""Step 3: vowel / consonant features, computed on the 'text' column.

Real names are pronounceable: they mix vowels and consonants in a natural way
('google', 'weather'). Random DGA names often have long blocks of consonants
('xkqjzpwv'), while some DGA families alternate vowels and consonants on
purpose to look pronounceable ('sumiwgecoll').

Only letters count: digits and '-' are ignored. 'y' is treated as a consonant.
Like the basic features, these need no training (no leakage risk).
"""
import re

import pandas as pd

VOWELS = set("aeiou")


def _letter_runs(s: str, pattern: str) -> int:
    """Length of the longest block matching pattern ('[aeiou]+' or consonants)."""
    return max((len(m) for m in re.findall(pattern, s)), default=0)


def _alternation(s: str) -> float:
    """Share of neighbouring letters that switch between vowel and consonant.

    'banana' -> b-a, a-n, n-a, a-n, n-a: all switch -> 1.0
    'strength' -> s-t, t-r, r-e (switch), e-n (switch), n-g, g-t, t-h -> 2/7
    Digits and '-' are removed first. Names with fewer than 2 letters -> 0.
    """
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 2:
        return 0.0
    is_vowel = [c in VOWELS for c in letters]
    switches = sum(a != b for a, b in zip(is_vowel, is_vowel[1:]))
    return switches / (len(letters) - 1)


def vowel_features(text: pd.Series) -> pd.DataFrame:
    """Compute vowel / consonant features for every string in 'text'.

    Returns a table with one row per domain and these columns:
      vowel_ratio             vowels / letters (0 if no letters)
      longest_consonant_run   longest block of consonants ('xkqjz' in 'axkqjzo' -> 5)
      longest_vowel_run       longest block of vowels ('aeiou' style runs)
      vc_alternation          share of neighbouring letters that switch
                              between vowel and consonant (see _alternation)
    """
    n_letters = text.str.count(r"[a-z]")
    n_vowels = text.str.count(r"[aeiou]")

    out = pd.DataFrame(index=text.index)
    out["vowel_ratio"] = (n_vowels / n_letters.clip(lower=1)).where(n_letters > 0, 0.0)
    # Consonants = letters that are not vowels (digits and '-' are not letters).
    out["longest_consonant_run"] = text.map(lambda s: _letter_runs(s, r"[b-df-hj-np-tv-z]+"))
    out["longest_vowel_run"] = text.map(lambda s: _letter_runs(s, r"[aeiou]+"))
    out["vc_alternation"] = text.map(_alternation)
    return out
