"""Linear probe on frozen embeddings with nested cross-validation.

For each outer fold:
1. Inner cross-validation on that fold's training set only scores every (layer, C) pair.
2. For **every** layer, a model with that layer's best C is trained on the full outer
   training set and predicts the outer test set -> a "layer -> quality" profile (exploratory).
3. The **headline prediction** uses the (layer, C) that won the inner loop. The outer test
   fold takes no part in that choice, so the headline number is honest.

Training always uses `X_fit` (e.g. clean or augmented audio), while predictions can be made
on several versions of the same clips (`X_eval`: telephone, noise, ...). Rows of all arrays
are the same clips in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from speech_affect.data.splits import inner_splits
from speech_affect.evaluation.metrics import uar

FIT = "fit"
"""Key of the predictions on the same condition the probe was trained on."""


def make_probe(C: float, max_iter: int) -> Pipeline:
    # StandardScaler: the L2 penalty treats all weights alike, so features must share a scale.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))


def _fit_score(X, y, tr, va, C, max_iter, n_classes) -> float:
    m = make_probe(C, max_iter).fit(X[tr], y[tr])
    return uar(y[va], m.predict(X[va]), n_classes)


def _fit_predict(X_tr, y_tr, X_tests: list[np.ndarray], C, max_iter) -> list[np.ndarray]:
    m = make_probe(C, max_iter).fit(X_tr, y_tr)
    return [m.predict(x) for x in X_tests]


@dataclass
class ProbeResult:
    pred: dict[str, np.ndarray]
    """Condition -> headline out-of-fold predictions `[N]` (layer and C from the inner loop)."""
    pred_by_layer: dict[str, np.ndarray]
    """Condition -> `[L, N]` predictions of every layer (C chosen for that layer)."""
    inner_scores: np.ndarray
    """`[n_outer, L, n_C]`: mean inner cross-validation UAR."""
    chosen: list[dict] = field(default_factory=list)


def nested_layer_probe(
    X_fit: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    outer_folds: np.ndarray,
    *,
    inner_grouped: bool,
    C_grid: tuple[float, ...],
    n_inner: int,
    max_iter: int,
    n_classes: int,
    seed: int,
    X_eval: dict[str, np.ndarray] | None = None,
    n_jobs: int = -2,
) -> ProbeResult:
    evals = {FIT: X_fit, **(X_eval or {})}
    names = list(evals)
    n, n_layers, _ = X_fit.shape
    n_outer = int(outer_folds.max()) + 1
    pred_by_layer = {c: np.full((n_layers, n), -1, dtype=int) for c in names}
    pred = {c: np.full(n, -1, dtype=int) for c in names}
    inner_scores = np.zeros((n_outer, n_layers, len(C_grid)))
    chosen = []
    par = Parallel(n_jobs=n_jobs)

    for k in range(n_outer):
        tr_mask = outer_folds != k
        te_mask = ~tr_mask
        y_tr, spk_tr = y[tr_mask], speakers[tr_mask]
        splits = inner_splits(y_tr, spk_tr, n_inner, seed + k, grouped=inner_grouped)
        layers = [
            np.ascontiguousarray(X_fit[tr_mask, li], dtype=np.float32) for li in range(n_layers)
        ]

        # 1. Inner loop: every (layer, C) on the outer fold's training set.
        grid = [
            (li, ci, si)
            for li in range(n_layers)
            for ci in range(len(C_grid))
            for si in range(len(splits))
        ]
        scores = par(
            delayed(_fit_score)(layers[li], y_tr, *splits[si], C_grid[ci], max_iter, n_classes)
            for li, ci, si in grid
        )
        acc = np.zeros((n_layers, len(C_grid)))
        for (li, ci, _), s in zip(grid, scores, strict=True):
            acc[li, ci] += s / len(splits)
        inner_scores[k] = acc

        # 2. Every layer with its best C -> predictions on all test conditions.
        best_c = acc.argmax(axis=1)
        tests = {c: evals[c][te_mask].astype(np.float32) for c in names}
        preds = par(
            delayed(_fit_predict)(
                layers[li], y_tr, [tests[c][:, li] for c in names], C_grid[best_c[li]], max_iter
            )
            for li in range(n_layers)
        )
        for li, per_cond in enumerate(preds):
            for c, p in zip(names, per_cond, strict=True):
                pred_by_layer[c][li, te_mask] = p

        # 3. Headline prediction: the best (layer, C) from the inner loop.
        best_layer, best_ci = np.unravel_index(acc.argmax(), acc.shape)
        for c in names:
            pred[c][te_mask] = pred_by_layer[c][best_layer, te_mask]
        chosen.append(
            {
                "fold": k,
                "layer": int(best_layer),
                "C": C_grid[best_ci],
                "inner_uar": float(acc[best_layer, best_ci]),
            }
        )

    for c in names:
        if (pred[c] < 0).any() or (pred_by_layer[c] < 0).any():
            raise RuntimeError(f"{c}: some clips received no out-of-fold prediction")
    return ProbeResult(pred, pred_by_layer, inner_scores, chosen)
