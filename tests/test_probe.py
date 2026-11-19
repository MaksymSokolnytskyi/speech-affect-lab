from __future__ import annotations

import numpy as np

from speech_affect.data.splits import speaker_folds
from speech_affect.evaluation.metrics import uar
from speech_affect.models.probe import FIT, nested_layer_probe


def test_nested_probe_finds_informative_layer():
    """Class information is present only in layer 2: the nested loop must pick it."""
    rng = np.random.default_rng(0)
    n_spk, per, n_layers, dim, K = 20, 30, 4, 8, 3
    speakers = np.repeat(np.arange(n_spk), per)
    y = rng.integers(0, K, len(speakers))
    X = rng.normal(size=(len(y), n_layers, dim)).astype(np.float16)
    X[:, 2, :K] += 3 * np.eye(K)[y]
    folds = speaker_folds(speakers, speakers % 2, n_folds=4, seed=0)
    noisy = X + rng.normal(scale=2.0, size=X.shape).astype(np.float16)

    res = nested_layer_probe(
        X,
        y,
        speakers,
        folds,
        inner_grouped=True,
        C_grid=(0.1, 1.0),
        n_inner=2,
        max_iter=500,
        n_classes=K,
        seed=0,
        X_eval={"noisy": noisy},
        n_jobs=1,
    )
    assert all(c["layer"] == 2 for c in res.chosen)
    assert uar(y, res.pred[FIT], K) > 0.9
    assert uar(y, res.pred["noisy"], K) < uar(y, res.pred[FIT], K)
    assert res.pred_by_layer[FIT].shape == (n_layers, len(y))
    assert res.inner_scores.shape == (4, n_layers, 2)
