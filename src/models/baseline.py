"""Baselines: majority class (Model 0) and logistic regression (Model 1).

Usage (from the repository root):
    python -m src.models.baseline [--setup binary|family] [--no-tld] [--seed N]
                                  [--sample N] [--test]

  --setup   binary: target DGA/legitimate; family: target legit or a DGA family,
            collapsed with P(DGA) = 1 - P(legit). Default: binary.
  --no-tld  remove the TLD (last label) from every domain (ablation).
  --sample  train on a random subset of N training rows (quick runs).
  --test    report on the test set; by default the validation set is used.

Logistic regression uses TF-IDF character 1-3-grams of the domain and the sample
weights from src/data/dataset.py. Metrics are computed on the DGA/legitimate decision.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data.dataset import load_splits


def strip_tld(domains):
    return domains.str.rsplit(".", n=1).str[0]


def binary_metrics(y_true, p_dga, threshold=0.5):
    y_pred = (p_dga >= threshold).astype(int)
    auc = roc_auc_score(y_true, p_dga) if len(np.unique(p_dga)) > 1 else 0.5
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "dga_precision": precision_score(y_true, y_pred, zero_division=0),
        "dga_recall": recall_score(y_true, y_pred),
        "roc_auc": auc,
    }


def recall_per_family(families, y_true, y_pred):
    dga = y_true == 1
    hit = pd.Series(y_pred[dga] == 1, index=families.to_numpy()[dga])
    return hit.groupby(level=0).mean().sort_values().round(3).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["binary", "family"], default="binary")
    ap.add_argument("--no-tld", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    train, val, test = load_splits()
    if args.sample:
        train = train.sample(n=args.sample, random_state=args.seed)
    evaluate = test if args.test else val
    x_train, x_eval = train["domain"], evaluate["domain"]
    if args.no_tld:
        x_train, x_eval = strip_tld(x_train), strip_tld(x_eval)
    y_eval = (evaluate["label"] == "dga").astype(int).to_numpy()
    target = train["label"] if args.setup == "binary" else train["family"]

    # Model 0: always predicts the most frequent training class.
    majority = train["label"].value_counts().idxmax()
    p_majority = np.full(len(evaluate), float(majority == "dga"))

    # Model 1: logistic regression on character n-grams.
    start = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=5, sublinear_tf=True)
    clf = LogisticRegression(max_iter=300, random_state=args.seed)
    clf.fit(vec.fit_transform(x_train), target, sample_weight=train["weight"])
    proba = clf.predict_proba(vec.transform(x_eval))
    legit_col = list(clf.classes_).index("legit")
    p_dga = 1 - proba[:, legit_col]
    seconds = time.time() - start

    y_pred = (p_dga >= 0.5).astype(int)
    report = {
        "setup": args.setup, "no_tld": args.no_tld, "seed": args.seed,
        "eval_set": "test" if args.test else "val", "train_rows": len(train),
        "fit_seconds": round(seconds),
        "model0_majority": binary_metrics(y_eval, p_majority),
        "model1_logreg": binary_metrics(y_eval, p_dga),
        "model1_recall_per_family": recall_per_family(evaluate["family"], y_eval, y_pred),
    }
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
