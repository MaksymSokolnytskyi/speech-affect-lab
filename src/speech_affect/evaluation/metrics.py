"""Classification metrics computed from a confusion matrix.

Everything goes through the confusion matrix because that keeps the bootstrap fast: the
matrix of a resample is a weighted sum of per-speaker matrices (see bootstrap.py).
"""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """`C[i, j]` is the number of clips of class i predicted as j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def uar_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Unweighted average recall (balanced accuracy). Also works on a batch `[..., K, K]`.

    A class absent from the sample is left out of the average.
    """
    return np.nanmean(recall_per_class(cm), axis=-1)


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    """Recall of each class (diagonal over row sums); NaN for a class with no clips."""
    support = cm.sum(axis=-1)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, diag / np.maximum(support, 1), np.nan)


def macro_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    diag = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    denom = support + predicted
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * diag / np.maximum(denom, 1), np.nan)
    return np.nanmean(f1, axis=-1)


def uar(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(uar_from_confusion(confusion(y_true, y_pred, n_classes)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(macro_f1_from_confusion(confusion(y_true, y_pred, n_classes)))
