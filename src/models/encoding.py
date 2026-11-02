"""Turn domain names and class names into integer arrays for the neural models.

The input is the full domain name as it is, TLD included (nothing is split off).
Each character becomes its ASCII code, as in Gregorio et al. (2024), and 0 is padding.
Domains are padded at the end to MAX_LEN = 93 characters, the longest domain in the
dataset (a carnavalheist domain under cloudapp.azure.com), so no domain is cut. The
paper used 70, the longest domain in its data.
"""
import numpy as np
import pandas as pd

MAX_LEN = 93
VOCAB = 256   # ASCII codes, 0 = padding


def encode(domains, max_len=MAX_LEN):
    """uint8 array of shape (n, max_len): ASCII codes, zero-padded at the end."""
    out = np.zeros((len(domains), max_len), dtype=np.uint8)
    for i, domain in enumerate(domains):
        if len(domain) > max_len:
            raise ValueError(f"domain longer than {max_len} characters: {domain}")
        codes = np.frombuffer(domain.encode("ascii"), dtype=np.uint8)
        out[i, :len(codes)] = codes
    return out


def class_codes(values, classes):
    """Index of each value in `classes` (int64), e.g. family names -> 0..103."""
    codes = pd.Categorical(values, categories=classes).codes
    if (codes < 0).any():
        raise ValueError(f"values not in classes: {set(values) - set(classes)}")
    return codes.astype(np.int64)
