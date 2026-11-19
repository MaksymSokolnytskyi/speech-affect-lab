"""Shared setup for the scripts in `experiments/`.

Every experiment needs the same few things: logging, integer labels and speaker groups from
the manifest, cached embeddings checked to be in manifest order, speaker-bootstrap helpers
bound to the evaluation config, and the probe hyperparameters. They live here so the scripts
contain only what is specific to each experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from speech_affect.config import Config
from speech_affect.data.cremad import load_manifest
from speech_affect.evaluation.bootstrap import (
    Interval,
    paired_speaker_bootstrap,
    speaker_bootstrap,
)
from speech_affect.features.extract import load_embeddings


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class Labels:
    manifest: pd.DataFrame
    emotions: list[str]
    y: np.ndarray
    """`[N]` integer emotion labels, indices into `emotions`."""
    speakers: np.ndarray
    sex: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.emotions)

    @property
    def chance(self) -> float:
        """UAR of a constant or random classifier."""
        return 1 / self.n_classes

    def encode(self, codes: pd.Series) -> np.ndarray:
        """Emotion codes ("ANG", ...) -> integer labels (NaN stays NaN)."""
        return codes.map({e: i for i, e in enumerate(self.emotions)}).to_numpy()


def load_labels(cfg: Config) -> Labels:
    man = load_manifest(cfg)
    emotions = list(cfg.cremad.emotions)
    y = man["emotion"].map({e: i for i, e in enumerate(emotions)}).to_numpy()
    return Labels(man, emotions, y, man["speaker_id"].to_numpy(), man["sex"].to_numpy())


def check_clip_order(clip_ids: np.ndarray, manifest: pd.DataFrame, what: str) -> None:
    """Raise if `clip_ids` are not exactly the manifest's clips in manifest order."""
    if not np.array_equal(np.asarray(clip_ids), manifest["clip_id"].to_numpy()):
        raise RuntimeError(f"{what}: clip order differs from the manifest")


def load_aligned_embeddings(
    model: str, condition: str, manifest: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Cached `[N, layers, dim]` embeddings and their metadata, checked against the manifest."""
    clip_ids, X, meta = load_embeddings(model, condition)
    check_clip_order(clip_ids, manifest, f"{model}/{condition} embeddings")
    return X, meta


@dataclass(frozen=True)
class Bootstrappers:
    """Speaker-bootstrap helpers bound to the labels, speakers and evaluation config."""

    y: np.ndarray
    speakers: np.ndarray
    n_classes: int
    n_boot: int
    seed: int
    ci_level: float

    def boot(
        self, pred: np.ndarray, mask: np.ndarray | None = None, seed_offset: int = 0
    ) -> tuple[Interval, np.ndarray]:
        """UAR interval and bootstrap replicates of `pred`, optionally on a subset `mask`.

        Use distinct `seed_offset`s for subsets whose replicates are later compared as
        independent (e.g. female vs male speakers).
        """
        m = np.ones(len(self.y), bool) if mask is None else mask
        return speaker_bootstrap(
            self.y[m],
            pred[m],
            self.speakers[m],
            self.n_classes,
            self.n_boot,
            self.seed + seed_offset,
            self.ci_level,
        )

    def interval(self, pred: np.ndarray, mask: np.ndarray | None = None) -> Interval:
        return self.boot(pred, mask)[0]

    def paired(self, pred_a: np.ndarray, pred_b: np.ndarray) -> Interval:
        """Paired interval of UAR(a) - UAR(b) on the same clips."""
        return paired_speaker_bootstrap(
            self.y,
            pred_a,
            pred_b,
            self.speakers,
            self.n_classes,
            self.n_boot,
            self.seed,
            self.ci_level,
        )


def make_bootstrappers(cfg: Config, y: np.ndarray, speakers: np.ndarray) -> Bootstrappers:
    ev = cfg.eval
    return Bootstrappers(
        y, speakers, len(cfg.cremad.emotions), ev.n_bootstrap, cfg.seed, ev.ci_level
    )


def probe_kwargs(cfg: Config, inner_grouped: bool = True) -> dict:
    """Keyword arguments of `nested_layer_probe` shared by all experiments."""
    ev = cfg.eval
    return {
        "inner_grouped": inner_grouped,
        "C_grid": ev.logreg_C_grid,
        "n_inner": ev.n_inner_folds,
        "max_iter": ev.logreg_max_iter,
        "n_classes": len(cfg.cremad.emotions),
        "seed": cfg.seed,
    }
