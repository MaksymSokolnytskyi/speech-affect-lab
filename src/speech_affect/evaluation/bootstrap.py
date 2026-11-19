"""Speaker-level bootstrap.

Clips of the same speaker are not independent, so we resample **speakers** with replacement.

Speed trick: compute each speaker's confusion matrix `[S, K, K]` once. A bootstrap resample
is then a vector "how many times each speaker was drawn" `w [S]`, and the resample's matrix
is `sum_s w_s * C_s`. A thousand replicates are a single tensor contraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from speech_affect.evaluation.metrics import confusion, uar_from_confusion

MetricFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def per_speaker_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, speakers: np.ndarray, n_classes: int
) -> np.ndarray:
    """`[S, K, K]` confusion matrices, one per speaker (in sorted speaker order)."""
    return np.stack(
        [
            confusion(y_true[speakers == s], y_pred[speakers == s], n_classes)
            for s in np.unique(speakers)
        ]
    )


def _speaker_weights(n_speakers: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """`[B, S]`: how many times each speaker was drawn into each bootstrap resample."""
    return rng.multinomial(n_speakers, np.full(n_speakers, 1 / n_speakers), size=n_boot)


def _interval(point: float, samples: np.ndarray, ci: float) -> Interval:
    a = (1 - ci) / 2
    lo, hi = np.nanquantile(samples, [a, 1 - a])
    return Interval(float(point), float(lo), float(hi), len(samples))


def speaker_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: np.ndarray,
    n_classes: int,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    metric: MetricFn = uar_from_confusion,
) -> tuple[Interval, np.ndarray]:
    """Percentile interval of the metric; also returns the bootstrap replicates themselves."""
    cms = per_speaker_confusion(y_true, y_pred, speakers, n_classes)
    w = _speaker_weights(len(cms), n_boot, np.random.default_rng(seed))
    samples = metric(np.einsum("bs,skl->bkl", w, cms))
    return _interval(float(metric(cms.sum(0))), samples, ci), samples


def paired_speaker_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    speakers: np.ndarray,
    n_classes: int,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    metric: MetricFn = uar_from_confusion,
) -> Interval:
    """Interval for the difference metric(A) - metric(B) on the same clips.

    In every replicate both systems are scored on the same resample of speakers, so the
    shared "difficulty" of the speakers cancels out and the interval is narrower.
    """
    cm_a = per_speaker_confusion(y_true, pred_a, speakers, n_classes)
    cm_b = per_speaker_confusion(y_true, pred_b, speakers, n_classes)
    w = _speaker_weights(len(cm_a), n_boot, np.random.default_rng(seed))
    diff = metric(np.einsum("bs,skl->bkl", w, cm_a)) - metric(np.einsum("bs,skl->bkl", w, cm_b))
    point = float(metric(cm_a.sum(0)) - metric(cm_b.sum(0)))
    return _interval(point, diff, ci)


def independent_difference(
    samples_a: np.ndarray, samples_b: np.ndarray, point_a: float, point_b: float, ci: float = 0.95
) -> Interval:
    """Difference between independent groups (e.g. female - male: disjoint speakers).

    The two sample arrays must come from bootstraps with *different* seeds; otherwise the
    replicates are correlated and the interval is wrong.
    """
    return _interval(point_a - point_b, samples_a - samples_b, ci)
