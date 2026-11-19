"""Cross-validation folds.

- `speaker_folds`: the main one. No speaker appears in both train and test. Stratified by
  sex (each speaker has one), so every fold has a similar male/female ratio.
- `random_folds`: only for demonstrating leakage; one speaker's clips are spread over folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def speaker_folds(speakers: np.ndarray, strata: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Test-fold index for every row; groups are speakers, stratification is `strata`."""
    folds = np.full(len(speakers), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(cv.split(np.zeros(len(speakers)), strata, groups=speakers)):
        folds[test] = k
    return folds


def random_folds(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Random folds stratified by label; speakers are NOT taken into account."""
    folds = np.full(len(labels), -1, dtype=int)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        folds[test] = k
    return folds


def inner_splits(
    labels: np.ndarray, speakers: np.ndarray, n_folds: int, seed: int, grouped: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner folds for hyperparameter selection within an outer training set.

    `grouped=True` splits by speaker (the honest setting). `False` splits at random (the
    leakage demo), so that hyperparameter selection also "sees" the same speakers, as in the
    typical mistake.
    """
    x = np.zeros(len(labels))
    if grouped:
        return list(GroupKFold(n_splits=n_folds).split(x, labels, groups=speakers))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(x, labels))
