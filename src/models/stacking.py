"""Stacking: use the family-first model's output as features for a binary model.

Two levels:
  Level 1  family model    features -> P(legit), P(family 1), ..., P(family 103)
  Level 2  binary model    [features + family probabilities] -> DGA / legit

Compared methods (same features, same training sample, same evaluation set):
  binary           level 2 alone: binary model on the features (as in notebook 04)
  family           level 1 alone: P(DGA) = 1 - P(legit) (family-first, as in notebook 04)
  stack_probs      binary model on the family probabilities only
  stack_both       binary model on the features + the family probabilities

Leakage rule (as for the n-gram features): the family probabilities of a TRAINING row
must come from a family model that did not see that row. Otherwise they are too good on
the training rows, and the binary model learns to trust them too much. So:
  - training rows: out-of-fold. The training sample is cut into n_folds parts; each
    part gets its probabilities from a family model trained on the other parts.
  - evaluation rows: family model trained on the whole training sample.
The probabilities are given to level 2 as log-probabilities (log P), which suit a
logistic regression better than raw probabilities (they are near 0 for most classes).

    from src.models.stacking import run_stacking, summarize_stacking
    table = run_stacking(feats, train_full, evaluate, X_eval_all, seeds=range(3))
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from src.models.evaluate import binary_metrics, p_dga_from_proba, recall_per_family
from src.models.logreg import LogReg

METHODS = ["binary", "family", "stack_probs", "stack_both"]
METRICS = ["macro_f1", "dga_precision", "dga_recall", "roc_auc"]


def _family_proba(X_fit, family, weight, X_pred, classes, l2):
    """Train a family model on X_fit, return its probabilities on X_pred.

    The columns follow `classes` (all families of the training sample): a family that
    is missing from X_fit (possible for tiny families in one fold) gets a tiny probability.
    """
    model = LogReg(l2=l2).fit(X_fit, family, sample_weight=weight)
    proba = np.full((len(X_pred), len(classes)), 1e-9)
    cols = np.searchsorted(classes, model.classes_)
    proba[:, cols] = model.predict_proba(X_pred)
    return proba / proba.sum(axis=1, keepdims=True)


def family_probabilities(X, family, weight, X_eval, n_folds=5, l2=1e-4, seed=0, n_jobs=None):
    """Level 1: out-of-fold probabilities on the training rows, normal ones on X_eval.

    Returns (P_train, P_eval, classes); P_train row i comes from a model that never saw row i.
    """
    X, X_eval = np.asarray(X, float), np.asarray(X_eval, float)
    family, weight = np.asarray(family), np.asarray(weight, float)
    classes = np.unique(family)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))

    jobs = [delayed(_family_proba)(X[fit], family[fit], weight[fit], X[held], classes, l2)
            for fit, held in folds]
    jobs.append(delayed(_family_proba)(X, family, weight, X_eval, classes, l2))
    n_jobs = n_jobs or min(os.cpu_count() or 1, len(jobs))
    out = Parallel(n_jobs=n_jobs)(jobs)

    P_train = np.zeros((len(X), len(classes)))
    for (_, held), proba in zip(folds, out[:-1]):
        P_train[held] = proba
    return P_train, out[-1], classes


def _test(y_eval, families, p_dga):
    recall = recall_per_family(families, y_eval, (p_dga >= 0.5).astype(int))
    return binary_metrics(y_eval, p_dga), recall, np.asarray(p_dga, dtype="float32")


def run_one(X, train, X_eval, evaluate, n_folds=5, l2=1e-4, seed=0, n_jobs=None):
    """All four methods for one training sample and one set of columns."""
    y_eval = (evaluate["label"] == "dga").astype(int).to_numpy()
    families = evaluate["family"]
    label, weight = train["label"], train["weight"]

    # Level 1: family probabilities (out-of-fold on train)
    P_train, P_eval, classes = family_probabilities(
        X, train["family"], weight, X_eval, n_folds, l2, seed, n_jobs)
    logP_train, logP_eval = np.log(P_train), np.log(P_eval)

    results = {}
    # Reference 1: binary model on the features
    model = LogReg(l2=l2).fit(X, label, sample_weight=weight)
    results["binary"] = _test(y_eval, families,
                              p_dga_from_proba(model.predict_proba(X_eval), model.classes_))
    # Reference 2: family-first, P(DGA) = 1 - P(legit)
    results["family"] = _test(y_eval, families, p_dga_from_proba(P_eval, classes))
    # Level 2a: binary model on the family probabilities only
    model = LogReg(l2=l2).fit(logP_train, label, sample_weight=weight)
    results["stack_probs"] = _test(
        y_eval, families, p_dga_from_proba(model.predict_proba(logP_eval), model.classes_))
    # Level 2b: binary model on the features + the family probabilities
    both_train = np.hstack([np.asarray(X, float), logP_train])
    both_eval = np.hstack([np.asarray(X_eval, float), logP_eval])
    model = LogReg(l2=l2).fit(both_train, label, sample_weight=weight)
    results["stack_both"] = _test(
        y_eval, families, p_dga_from_proba(model.predict_proba(both_eval), model.classes_))
    return results


def summarize_stacking(table, metric="macro_f1"):
    """Mean / std over seeds per method, and the gains of the stacked models.

    Gains are computed seed by seed, then averaged. 'better' follows the proposal's
    rule: mean gain larger than its standard deviation.
    """
    values = table[metric].astype(float).unstack(["TLD", "method"])
    summary = pd.DataFrame({"mean": values.mean(), "std": values.std()})

    gains = {}
    for tld in values.columns.get_level_values("TLD").unique():
        for new in ["stack_probs", "stack_both"]:
            for ref in ["binary", "family"]:
                g = values[tld, new] - values[tld, ref]
                gains[tld, f"{new} - {ref}"] = {
                    "mean_gain": g.mean(), "std_gain": g.std(),
                    "better": bool(g.mean() > g.std())}
    gains = pd.DataFrame(gains).T
    gains.index.names = ["TLD", "comparison"]
    return summary, gains


def _log_to_mlflow(table, recalls, experiment, run_name, settings):
    """One MLflow run per (seed, TLD, method), plus one summary run."""
    from src.models import tracking

    tracking.setup(experiment)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for (seed, tld, method), row in table.iterrows():
            with tracking.start_run(f"{run_name} {method} TLD={tld} seed={seed}",
                                    {**settings, "method": method, "tld": tld, "seed": seed}):
                tracking.log_metrics({m: row[m] for m in METRICS})
                path = tmp / "recall_per_family.csv"
                recalls[seed, tld, method].rename("recall").to_csv(path)
                tracking.log_files(path)

        summary, gains = summarize_stacking(table)
        metrics = {}
        for (tld, method), row in summary.iterrows():
            metrics[f"macro_f1_mean_{method}_tld_{tld}"] = row["mean"]
            metrics[f"macro_f1_std_{method}_tld_{tld}"] = row["std"]
        for (tld, comp), row in gains.iterrows():
            key = comp.replace(" - ", "_vs_")
            metrics[f"gain_{key}_tld_{tld}"] = row["mean_gain"]
            metrics[f"gain_std_{key}_tld_{tld}"] = row["std_gain"]
        with tracking.start_run(f"{run_name} summary", settings):
            tracking.log_metrics(metrics)
            files = {"all_runs.csv": table, "summary.csv": summary, "gains.csv": gains}
            for file_name, df in files.items():
                df.to_csv(tmp / file_name)
            tracking.log_files(*(tmp / f for f in files))


def run_stacking(feats, train_full, evaluate, X_eval_all, seeds=range(3), sample=100_000,
                 tlds=("yes", "no"), n_folds=5, l2=1e-4, n_jobs=None, track=True,
                 experiment="baselines", run_name="logreg_stacking", eval_set="val",
                 verbose=True, save_dir=None):
    """Run the four methods for every seed and TLD setting.

    Returns one row per (seed, TLD, method) with the metrics.
    save_dir: if given, also writes <run_name>_results.csv and
    <run_name>_predictions.npz (P(DGA) of every evaluation domain, per run).
    """
    name_cols, tld_cols = feats["name_cols"], feats["tld_cols"]
    rows, recalls, preds = {}, {}, {}
    for seed in seeds:
        train = train_full.sample(n=sample, random_state=seed) if sample else train_full
        for tld in tlds:
            cols = name_cols + tld_cols if tld == "yes" else name_cols
            if verbose:
                print(f"seed {seed} | TLD {tld} ...", flush=True)
            res = run_one(feats["train"].loc[train.index, cols], train,
                          X_eval_all[cols], evaluate, n_folds, l2, seed, n_jobs)
            for method, (metrics, recall, p) in res.items():
                rows[seed, tld, method] = metrics
                recalls[seed, tld, method] = recall
                preds[seed, tld, method] = p

    table = pd.DataFrame(rows).T
    table.index.names = ["seed", "TLD", "method"]
    table = table.sort_index()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_dir / f"{run_name}_results.csv")
        np.savez_compressed(save_dir / f"{run_name}_predictions.npz",
                            index=evaluate.index.to_numpy(),
                            y_true=(evaluate["label"] == "dga").astype(int).to_numpy(),
                            **{"|".join(map(str, k)): v for k, v in preds.items()})

    if track:
        settings = {"model": "logreg_stacking", "features": "handcrafted_74",
                    "sample": sample or "full", "l2": l2, "weights": True,
                    "n_folds": n_folds, "n_seeds": len(list(seeds)), "eval_set": eval_set}
        _log_to_mlflow(table, recalls, experiment, run_name, settings)
    return table
