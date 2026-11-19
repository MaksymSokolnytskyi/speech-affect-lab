"""Baseline: classic eGeMAPS features vs WavLM (criterion pre-registered in
docs/PREREGISTRATION.md).

Same logistic regression, same nested speaker-independent cross-validation, same folds as
stage 1 (taken from predictions.parquet of the latest exp01 run). Only C is selected: there is
a single "layer" of 88 features. The WavLM - eGeMAPS difference is a paired speaker bootstrap
on the same clips.

Requires the eGeMAPS features (`uv run python -m speech_affect.features.egemaps`):
    uv run python experiments/exp04_egemaps_baseline.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from speech_affect.config import get_config
from speech_affect.data.splits import speaker_folds
from speech_affect.evaluation import plots
from speech_affect.evaluation.metrics import confusion, recall_per_class
from speech_affect.experiment import (
    check_clip_order,
    load_labels,
    make_bootstrappers,
    probe_kwargs,
    setup_logging,
)
from speech_affect.features.egemaps import load_egemaps
from speech_affect.models.probe import FIT, nested_layer_probe
from speech_affect.runs import latest_run, new_run, write_json

log = logging.getLogger("exp04")


def main() -> None:
    setup_logging()
    cfg = get_config()
    lab = load_labels(cfg)
    man, y, spk, K = lab.manifest, lab.y, lab.speakers, lab.n_classes
    exp01 = latest_run("exp01_emotion_probe")
    if exp01 is None:
        raise FileNotFoundError("run experiments/exp01_emotion_probe.py first")
    run = new_run("exp04_egemaps_baseline", {"exp01_run": str(exp01)})
    log.info("run directory: %s", run)

    prev = pd.read_parquet(exp01 / "predictions.parquet")
    check_clip_order(prev["clip_id"].to_numpy(), man, "exp01 predictions")
    folds = prev["fold_speaker"].to_numpy()
    # Folds are deterministic given the seed; check that these really are the same folds.
    if not np.array_equal(folds, speaker_folds(spk, lab.sex, cfg.eval.n_outer_folds, cfg.seed)):
        raise RuntimeError("exp01 folds differ from the folds derived from the current config")

    feats = load_egemaps()
    check_clip_order(feats["clip_id"].to_numpy(), man, "eGeMAPS features")
    X = feats.drop(columns="clip_id").to_numpy(np.float64)
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        raise RuntimeError(f"{n_nan} NaN values in the features")
    log.info("eGeMAPS: %s", X.shape)

    res = nested_layer_probe(X[:, None, :], y, spk, folds, **probe_kwargs(cfg))
    p_ege = res.pred[FIT]
    p_wavlm = prev["pred_wavlm_speaker"].to_numpy()
    p_hubert = prev["pred_hubert_speaker"].to_numpy()
    bs = make_bootstrappers(cfg, y, spk)

    cm_ege = confusion(y, p_ege, K)
    summary = {
        "n_features": X.shape[1],
        "egemaps": {"uar": bs.interval(p_ege).as_dict(), "chosen": res.chosen},
        "wavlm": {"uar": bs.interval(p_wavlm).as_dict()},
        "hubert": {"uar": bs.interval(p_hubert).as_dict()},
        "wavlm_minus_egemaps": bs.paired(p_wavlm, p_ege).as_dict(),
        "hubert_minus_egemaps": bs.paired(p_hubert, p_ege).as_dict(),
        # Exploratory: where exactly the hand-crafted features fall behind.
        "recall_per_class": {
            "emotions": lab.emotions,
            "egemaps": recall_per_class(cm_ege).tolist(),
            "wavlm": recall_per_class(confusion(y, p_wavlm, K)).tolist(),
        },
    }
    pd.DataFrame({"clip_id": man["clip_id"], "pred_egemaps_speaker": p_ege}).to_parquet(
        run / "predictions.parquet", index=False
    )
    plots.confusion_matrix(
        cm_ege, lab.emotions, run / "confusion_egemaps.png", "eGeMAPS: speaker-independent split"
    )
    write_json(run / "summary.json", summary)  # last: marks the run as finished
    log.info("eGeMAPS UAR %s", summary["egemaps"]["uar"])
    log.info("WavLM - eGeMAPS %s", summary["wavlm_minus_egemaps"])


if __name__ == "__main__":
    main()
