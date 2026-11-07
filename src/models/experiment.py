"""Run Model 1 (logistic regression) over several seeds, in parallel.

Each seed draws a different random sample of the training set (as in
02_baseline.ipynb), then trains the four setups (binary / family-first x with /
without TLD) and tests them with the shared metrics. Repeating over seeds shows
how much the results vary, so a difference between setups can be compared with
its standard deviation (the rule in the proposal).

    from src.models.experiment import run_seeds
    results = run_seeds(feats, train_full, evaluate, seeds=range(5), sample=100_000)
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.models.evaluate import binary_metrics, p_dga_from_proba
from src.models.logreg import LogReg

SETUPS = [("binary", "yes"), ("binary", "no"), ("family", "yes"), ("family", "no")]


def _fit_and_test(X_train, target, weight, X_eval, y_eval, l2):
    """Steps 3 and 4 for one setup: fit, then metrics on the evaluation set."""
    model = LogReg(l2=l2).fit(X_train, target, sample_weight=weight)
    p_dga = p_dga_from_proba(model.predict_proba(X_eval), model.classes_)
    return {**binary_metrics(y_eval, p_dga), "iterations": model.n_iter_,
            "converged": model.converged_, "fit_s": round(model.fit_seconds_)}


def run_seeds(feats, train_full, evaluate, X_eval_all, seeds=range(5),
              sample=100_000, l2=1e-4, n_jobs=None):
    """Train and test the four setups for every seed.

    feats: output of build_features; train_full: full training split (with
    'weight'); evaluate / X_eval_all: the evaluation split and its features.
    Returns one row per (seed, setup, TLD) with the metrics.
    """
    name_cols, tld_cols = feats["name_cols"], feats["tld_cols"]
    y_eval = (evaluate["label"] == "dga").astype(int).to_numpy()

    jobs, keys = [], []
    for seed in seeds:
        train = train_full.sample(n=sample, random_state=seed) if sample else train_full
        X = feats["train"].loc[train.index]
        for setup, tld in SETUPS:
            cols = name_cols + tld_cols if tld == "yes" else name_cols
            target = train["label"] if setup == "binary" else train["family"]
            jobs.append(delayed(_fit_and_test)(X[cols], target, train["weight"],
                                               X_eval_all[cols], y_eval, l2))
            keys.append((seed, setup, tld))

    # Slowest jobs (family-first) first, so the cores stay busy.
    order = sorted(range(len(jobs)), key=lambda i: keys[i][1] != "family")
    n_jobs = n_jobs or min(os.cpu_count() or 1, len(jobs))
    out = Parallel(n_jobs=n_jobs)(jobs[i] for i in order)

    rows = {keys[i]: r for i, r in zip(order, out)}
    table = pd.DataFrame(rows).T
    table.index.names = ["seed", "setup", "TLD"]
    return table.sort_index()


def summarize(table, metric="macro_f1"):
    """Mean and standard deviation over seeds, and the family-first gain.

    gain = family-first minus binary, computed seed by seed (same sample),
    then averaged. 'family-first better' follows the proposal's rule:
    mean gain larger than its standard deviation.
    """
    values = table[metric].astype(float).unstack(["setup", "TLD"])
    summary = pd.DataFrame({"mean": values.mean(), "std": values.std()})

    gains = {}
    for tld in ["yes", "no"]:
        g = values["family", tld] - values["binary", tld]
        gains[tld] = {"mean_gain": g.mean(), "std_gain": g.std(),
                      "family_first_better": bool(g.mean() > g.std())}
    return summary, pd.DataFrame(gains).T.rename_axis("TLD")
