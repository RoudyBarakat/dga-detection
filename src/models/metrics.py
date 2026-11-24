"""Metrics on the DGA/legitimate decision, shared by every model.

y_true is 1 for DGA and 0 for legitimate; p_dga is the predicted P(DGA).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
    return hit.groupby(level=0).mean()


def errors_by_group(groups, y_true, predictions):
    """Missed DGA domains (false negatives) and false alarms (false positives) per group.

    groups: one value per domain (e.g. a band of name length; a categorical keeps its
    order). predictions: {model name: its 0/1 predictions}. Returns one row per group,
    plus the total: the number of DGA and of legitimate domains, then for each model
    the DGA domains it misses, then for each model the legitimate domains it calls DGA.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    dga = pd.Series(np.asarray(y_true) == 1)

    def count(mask):
        return mask.groupby(groups, observed=False, sort=True).sum()

    called = {name: pd.Series(np.asarray(y_pred) == 1) for name, y_pred in predictions.items()}
    table = pd.concat({
        "domains": pd.DataFrame({"DGA": count(dga), "legitimate": count(~dga)}),
        "missed DGA (FN)": pd.DataFrame({n: count(dga & ~c) for n, c in called.items()}),
        "false alarms (FP)": pd.DataFrame({n: count(~dga & c) for n, c in called.items()}),
    }, axis=1)
    table.loc["total"] = table.sum()
    return table
