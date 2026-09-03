"""Evaluation metrics, defined once so every arm is scored identically.

The metric choices here are the project's evaluation stance in code form. See
docs/METHODOLOGY.md §1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve — the headline metric.

    `average_precision_score` is used rather than auc(recall, precision) because the
    latter interpolates linearly between operating points, which is not meaningful
    for a PR curve and inflates the result. Average precision sums the actual
    precision at each threshold weighted by the recall it gains.
    """
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reported ONLY to demonstrate why it is the wrong metric here.

    With ~1,100 negatives per positive, tens of thousands of false alarms barely move
    the false-positive rate, so this stays flatteringly high while the alert queue is
    unusable. Showing both numbers side by side is more convincing than asserting it.
    """
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict:
    """Precision among the k highest-scoring items — the analyst-capacity view.

    A compliance team can review a fixed number of alerts per day. This asks: of the
    top k we hand them, how many are real? It is the number an operations lead cares
    about, and it is invariant to how the model scores everything below the cut.
    """
    k = min(k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    hits = int(y_true[top].sum())
    total_positives = int(y_true.sum())
    return {
        "k": k,
        "hits": hits,
        "precision": hits / k if k else 0.0,
        "recall": hits / total_positives if total_positives else 0.0,
    }


def alerts_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> dict:
    """How many alerts must be reviewed to catch `target_recall` of laundering.

    This is the business-facing number that replaces the un-computable
    "estimated analyst-time reduction" from the original plan: it is measured, not
    assumed, and it is directly comparable across arms.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns arrays one longer than thresholds; drop the
    # trailing point, which corresponds to recall=0 with no threshold.
    recall, precision = recall[:-1], precision[:-1]

    viable = recall >= target_recall
    if not viable.any():
        return {"target_recall": target_recall, "achievable": False}

    # Among thresholds meeting the recall target, take the most precise one — that is
    # the smallest alert queue that still hits the target.
    idx = int(np.argmax(np.where(viable, precision, -1)))
    n_positives = int(y_true.sum())
    n_alerts = int(round(n_positives * recall[idx] / precision[idx])) if precision[idx] > 0 else 0
    return {
        "target_recall": target_recall,
        "achievable": True,
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "alerts_to_review": n_alerts,
    }


def summarise(y_true: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]) -> dict:
    """Full metric bundle for one arm on one split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype="float64")
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "precision_at_k": {str(k): precision_at_k(y_true, scores, k) for k in ks},
        "alerts_at_80_recall": alerts_for_recall(y_true, scores, 0.80),
    }
