"""Run Model 1 (logistic regression) over several seeds, in parallel, tracked in MLflow.

Each seed draws a different random sample of the training set (as in
02_baseline.ipynb), then trains the four setups (binary / family-first x with /
without TLD) and tests them with the shared metrics. Repeating over seeds shows
how much the results vary, so a difference between setups can be compared with
its standard deviation (the rule in the proposal).

With track=True, everything is logged in MLflow with the team's tracking module
(src/models/tracking.py), in the experiment "baselines": one run per fit
(seed x setup x TLD) and one summary run (mean / std and family-first gains).

    from src.models.experiment import run_seeds, summarize
    results = run_seeds(feats, train_full, evaluate, X_eval_all, seeds=range(5))
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.models.evaluate import binary_metrics, p_dga_from_proba, recall_per_family
from src.models.logreg import LogReg

SETUPS = [("binary", "yes"), ("binary", "no"), ("family", "yes"), ("family", "no")]
METRICS = ["macro_f1", "dga_precision", "dga_recall", "roc_auc"]


def _fit_and_test(X_train, target, weight, X_eval, y_eval, families, l2):
    """Steps 3 and 4 for one setup: fit, then metrics on the evaluation set."""
    model = LogReg(l2=l2).fit(X_train, target, sample_weight=weight)
    p_dga = p_dga_from_proba(model.predict_proba(X_eval), model.classes_)
    recall = recall_per_family(families, y_eval, (p_dga >= 0.5).astype(int))
    metrics = {**binary_metrics(y_eval, p_dga), "iterations": model.n_iter_,
               "converged": model.converged_, "fit_s": round(model.fit_seconds_)}
    return metrics, recall


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


def _log_to_mlflow(table, recalls, experiment, run_name, settings):
    """Log the experiment with the team's tracking module (src/models/tracking.py).

    One run per fit (seed x setup x TLD) with its settings, metrics and recall per
    family, plus one summary run with the mean / std per setup and the family-first
    gains (the numbers of the proposal's rule).
    """
    from src.models import tracking

    tracking.setup(experiment)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # One run per fit
        for (seed, s, tld), row in table.iterrows():
            name = f"{run_name} {s} TLD={tld} seed={seed}"
            with tracking.start_run(name, {**settings, "setup": s, "tld": tld,
                                           "seed": seed}):
                tracking.log_metrics({m: row[m] for m in METRICS + ["fit_s", "iterations"]})
                path = tmp / "recall_per_family.csv"
                recalls[seed, s, tld].rename("recall").to_csv(path)
                tracking.log_files(path)

        # One summary run: mean / std over the seeds and the family-first gains
        summary, gains = summarize(table)
        metrics = {}
        for (s, tld), row in summary.iterrows():
            metrics[f"macro_f1_mean_{s}_tld_{tld}"] = row["mean"]
            metrics[f"macro_f1_std_{s}_tld_{tld}"] = row["std"]
        for tld, row in gains.iterrows():
            metrics[f"gain_tld_{tld}"] = row["mean_gain"]
            metrics[f"gain_std_tld_{tld}"] = row["std_gain"]
        with tracking.start_run(f"{run_name} summary", settings):
            tracking.log_metrics(metrics)
            files = {"all_runs.csv": table, "summary.csv": summary,
                     "family_first_gain.csv": gains}
            for file_name, df in files.items():
                df.to_csv(tmp / file_name)
            tracking.log_files(*(tmp / f for f in files))


def run_seeds(feats, train_full, evaluate, X_eval_all, seeds=range(5),
              sample=100_000, l2=1e-4, n_jobs=None, track=True,
              experiment="baselines", run_name="logreg_handcrafted", eval_set="val"):
    """Train and test the four setups for every seed.

    feats: output of build_features; train_full: full training split (with
    'weight'); evaluate / X_eval_all: the evaluation split and its features.
    track: log everything in MLflow with the team's tracking module
    (src/models/tracking.py), in the MLflow experiment `experiment`.
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
                                               X_eval_all[cols], y_eval,
                                               evaluate["family"], l2))
            keys.append((seed, setup, tld))

    # Slowest jobs (family-first) first, so the cores stay busy.
    order = sorted(range(len(jobs)), key=lambda i: keys[i][1] != "family")
    n_jobs = n_jobs or min(os.cpu_count() or 1, len(jobs))
    out = Parallel(n_jobs=n_jobs)(jobs[i] for i in order)

    rows = {keys[i]: metrics for i, (metrics, _) in zip(order, out)}
    recalls = {keys[i]: recall for i, (_, recall) in zip(order, out)}
    table = pd.DataFrame(rows).T
    table.index.names = ["seed", "setup", "TLD"]
    table = table.sort_index()

    if track:
        settings = {"model": "logreg", "features": "handcrafted_74",
                    "sample": sample or "full", "l2": l2, "weights": True,
                    "n_seeds": len(list(seeds)), "eval_set": eval_set}
        _log_to_mlflow(table, recalls, experiment, run_name, settings)
    return table
