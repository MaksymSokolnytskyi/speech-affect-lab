"""Stage 2. What do the telephone channel and noise do to the model, and does augmentation help?

Setting A: the probe is trained on CLEAN audio and tested on every condition.
Setting B: the probe is trained on AUGMENTED audio (each training clip gets a random condition
with noise from the train pool) and tested on the same conditions with test-pool noise.

Everything uses the same speaker folds as stage 1. Test conditions are the same clips, so all
comparisons are paired.

Requires embeddings for all conditions:
    uv run python experiments/exp02_telephone.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from speech_affect.channel.conditions import TEST_CONDITIONS
from speech_affect.config import get_config
from speech_affect.data.splits import speaker_folds
from speech_affect.evaluation import plots
from speech_affect.experiment import (
    load_aligned_embeddings,
    load_labels,
    make_bootstrappers,
    probe_kwargs,
    setup_logging,
)
from speech_affect.models.probe import FIT, nested_layer_probe
from speech_affect.runs import new_run, write_json

log = logging.getLogger("exp02")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Stage 2: telephone channel, noise, augmentation.")
    ap.add_argument("--models", nargs="+", default=["wavlm", "hubert"])
    args = ap.parse_args()

    cfg = get_config()
    lab = load_labels(cfg)
    man, y, spk = lab.manifest, lab.y, lab.speakers
    run = new_run("exp02_telephone", {"models": args.models, "conditions": TEST_CONDITIONS})
    log.info("run directory: %s", run)

    folds = speaker_folds(spk, lab.sex, cfg.eval.n_outer_folds, cfg.seed)
    bs = make_bootstrappers(cfg, y, spk)
    iv, paired = bs.interval, bs.paired
    common = probe_kwargs(cfg)
    degraded = [c for c in TEST_CONDITIONS if c != "clean"]

    summary: dict = {"conditions": TEST_CONDITIONS, "chance": lab.chance}
    for model in args.models:
        X = {c: load_aligned_embeddings(model, c, man)[0] for c in TEST_CONDITIONS}
        X_aug, _ = load_aligned_embeddings(model, "aug_train", man)

        log.info("%s: setting A (trained on clean)", model)
        res_a = nested_layer_probe(
            X["clean"], y, spk, folds, X_eval={c: X[c] for c in degraded}, **common
        )
        pred_a = {"clean": res_a.pred[FIT], **{c: res_a.pred[c] for c in degraded}}
        log.info("%s: setting B (trained with augmentation)", model)
        res_b = nested_layer_probe(X_aug, y, spk, folds, X_eval=X, **common)
        pred_b = {c: res_b.pred[c] for c in TEST_CONDITIONS}

        s: dict = {
            "A_train_clean": {},
            "B_train_aug": {},
            "drop_clean_minus_cond_A": {},
            "gain_B_minus_A": {},
            "best_layer_by_condition_A": {},
            "chosen_A": res_a.chosen,
            "chosen_B": res_b.chosen,
        }
        for c in TEST_CONDITIONS:
            s["A_train_clean"][c] = iv(pred_a[c]).as_dict()
            s["B_train_aug"][c] = iv(pred_b[c]).as_dict()
            s["gain_B_minus_A"][c] = paired(pred_b[c], pred_a[c]).as_dict()
            if c != "clean":
                s["drop_clean_minus_cond_A"][c] = paired(pred_a["clean"], pred_a[c]).as_dict()
            # Does the best layer shift under degraded conditions (setting A profile)?
            byl = res_a.pred_by_layer[FIT if c == "clean" else c]
            layer_uar = [iv(byl[li]).point for li in range(byl.shape[0])]
            s["best_layer_by_condition_A"][c] = {
                "layer": int(np.argmax(layer_uar)),
                "uar_by_layer": layer_uar,
            }
        summary[model] = s

        series = {
            name: {k: [s[key][c][k] for c in TEST_CONDITIONS] for k in ("point", "lo", "hi")}
            for key, name in (
                ("A_train_clean", "trained on clean (A)"),
                ("B_train_aug", "trained with augmentation (B)"),
            )
        }
        plots.condition_dots(
            series,
            [plots.CONDITION_TICKS[c] for c in TEST_CONDITIONS],
            lab.chance,
            run / f"conditions_{model}.png",
            f"{plots.LABELS[model]}: UAR by test condition (95% interval)",
        )
        np.savez_compressed(
            run / f"preds_{model}.npz",
            **{f"A_{c}": pred_a[c] for c in TEST_CONDITIONS},
            **{f"B_{c}": pred_b[c] for c in TEST_CONDITIONS},
        )
        for c in TEST_CONDITIONS:
            log.info("%s %-10s A %s | B %s", model, c, iv(pred_a[c]), iv(pred_b[c]))

    write_json(run / "summary.json", summary)
    log.info("done: %s", run)


if __name__ == "__main__":
    main()
