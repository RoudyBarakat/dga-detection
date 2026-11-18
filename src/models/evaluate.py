"""Turn a model's class probabilities into P(DGA), for both setups.

The metrics themselves (binary_metrics, recall_per_family) are in
src/models/metrics.py, shared by every model.
"""


def p_dga_from_proba(proba, classes):
    """Collapse class probabilities to P(DGA) = 1 - P(legit).

    Works for both setups: binary (classes 'dga', 'legit') and family-first
    (classes 'legit' + one per DGA family).
    """
    return 1 - proba[:, list(classes).index("legit")]
