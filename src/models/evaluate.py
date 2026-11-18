"""Metrics on the DGA/legitimate decision, shared by every model.

binary_metrics and recall_per_family are the functions from
notebooks/02_baseline.ipynb, moved here so every notebook uses the same code.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true, p_dga, threshold=0.5):
    """Macro-F1, DGA precision, DGA recall and ROC-AUC.

    y_true: 1 = DGA, 0 = legit. p_dga: predicted probability of DGA.
    A domain is called DGA when p_dga >= threshold.
    """
    y_pred = (p_dga >= threshold).astype(int)
    auc = roc_auc_score(y_true, p_dga) if len(np.unique(p_dga)) > 1 else 0.5
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "dga_precision": precision_score(y_true, y_pred, zero_division=0),
        "dga_recall": recall_score(y_true, y_pred),
        "roc_auc": auc,
    }


def recall_per_family(families, y_true, y_pred):
    """Share of each DGA family's domains that are called DGA."""
    dga = y_true == 1
    hit = pd.Series(y_pred[dga] == 1, index=families.to_numpy()[dga])
    return hit.groupby(level=0).mean()


def p_dga_from_proba(proba, classes):
    """Collapse class probabilities to P(DGA) = 1 - P(legit).

    Works for both setups: binary (classes 'dga', 'legit') and family-first
    (classes 'legit' + one per DGA family).
    """
    return 1 - proba[:, list(classes).index("legit")]
