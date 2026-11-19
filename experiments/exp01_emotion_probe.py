"""Stage 1. How well do frozen WavLM and HuBERT recognise emotion, measured honestly?

Questions this script answers:
1. Headline: UAR on a speaker-independent split (layer and C chosen in the nested loop).
2. Leakage: how much a random split inflates UAR (paired bootstrap of the difference).
3. Layer profile: UAR of every layer on its own.
4. WavLM vs HuBERT (paired difference on the same clips).
5. Slices: sex, intensity (sentence IEO only), comparison with the listener votes.

Requires `clean` embeddings for both models:
    uv run python experiments/exp01_emotion_probe.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from speech_affect.config import get_config
from speech_affect.data.splits import random_folds, speaker_folds
from speech_affect.evaluation import plots
from speech_affect.evaluation.bootstrap import independent_difference, speaker_bootstrap
from speech_affect.evaluation.metrics import confusion, macro_f1_from_confusion
from speech_affect.experiment import (
    load_aligned_embeddings,
    load_labels,
    make_bootstrappers,
    probe_kwargs,
    setup_logging,
)
from speech_affect.models.probe import FIT, nested_layer_probe
from speech_affect.runs import new_run, write_json

log = logging.getLogger("exp01")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Stage 1: emotion probe on clean audio.")
    ap.add_argument("--models", nargs="+", default=["wavlm", "hubert"])
    ap.add_argument("--condition", default="clean")
    args = ap.parse_args()

    cfg = get_config()
    ev = cfg.eval
    lab = load_labels(cfg)
    man, y, spk, sex, K = lab.manifest, lab.y, lab.speakers, lab.sex, lab.n_classes
    run = new_run("exp01_emotion_probe", {"models": args.models, "condition": args.condition})
    log.info("run directory: %s", run)

    folds = {
        "speaker": speaker_folds(spk, sex, ev.n_outer_folds, cfg.seed),
        "random": random_folds(y, ev.n_outer_folds, cfg.seed),
    }
    bs = make_bootstrappers(cfg, y, spk)
    boot = bs.boot

    preds = man[
        ["clip_id", "speaker_id", "sex", "emotion", "sentence", "intensity", "voice_vote_emotion"]
    ].copy()
    preds["fold_speaker"], preds["fold_random"] = folds["speaker"], folds["random"]
    summary: dict = {"n_clips": len(y), "n_speakers": len(np.unique(spk)), "chance": lab.chance}
    profiles, random_profiles, headline = {}, {}, {}

    for model in args.models:
        X, meta = load_aligned_embeddings(model, args.condition, man)
        res = {}
        for regime, f in folds.items():
            log.info("%s: %s split", model, regime)
            res[regime] = nested_layer_probe(
                X, y, spk, f, **probe_kwargs(cfg, inner_grouped=(regime == "speaker"))
            )
            preds[f"pred_{model}_{regime}"] = res[regime].pred[FIT]
            np.savez_compressed(
                run / f"by_layer_{model}_{regime}.npz",
                pred_by_layer=res[regime].pred_by_layer[FIT],
                inner_scores=res[regime].inner_scores,
            )

        s: dict = {"encoder": meta}
        for regime in folds:
            iv, _ = boot(res[regime].pred[FIT])
            cm = confusion(y, res[regime].pred[FIT], K)
            s[regime] = {
                "uar": iv.as_dict(),
                "macro_f1": float(macro_f1_from_confusion(cm)),
                "chosen": res[regime].chosen,
            }
        s["leak_random_minus_speaker"] = bs.paired(
            res["random"].pred[FIT], res["speaker"].pred[FIT]
        ).as_dict()

        # Layer profile (exploratory, not the headline number).
        prof = {"point": [], "lo": [], "hi": []}
        for li in range(X.shape[1]):
            iv, _ = boot(res["speaker"].pred_by_layer[FIT][li])
            for k in prof:
                prof[k].append(getattr(iv, k))
        profiles[model] = {k: np.array(v) for k, v in prof.items()}
        random_profiles[model] = np.array(
            [bs.interval(res["random"].pred_by_layer[FIT][li]).point for li in range(X.shape[1])]
        )
        s["layer_profile_speaker"] = {k: [float(v) for v in vals] for k, vals in prof.items()}
        s["layer_profile_random"] = random_profiles[model].tolist()

        # Sex slice. The speakers differ, so the difference is between independent groups:
        # each group gets its own bootstrap seed, or the replicates would be correlated.
        p = res["speaker"].pred[FIT]
        by_sex = {
            sx: boot(p, sex == sx, seed_offset=1 + i) for i, sx in enumerate(("Female", "Male"))
        }
        s["by_sex"] = {sx: iv.as_dict() for sx, (iv, _) in by_sex.items()}
        s["by_sex"]["female_minus_male"] = independent_difference(
            by_sex["Female"][1],
            by_sex["Male"][1],
            by_sex["Female"][0].point,
            by_sex["Male"][0].point,
            ev.ci_level,
        ).as_dict()

        # Intensity: LO/MD/HI exist only for sentence IEO and only for non-neutral emotions.
        is_ieo = (man["sentence"] == "IEO").to_numpy()
        s["by_intensity_IEO"] = {
            lvl: bs.interval(p, is_ieo & (man["intensity"] == lvl).to_numpy()).as_dict()
            for lvl in ("LO", "MD", "HI")
        }

        # Listeners: on clips without a tie, compare the model and the listener majority
        # against the intended label.
        vote = man["voice_vote_emotion"]
        has_vote = vote.notna().to_numpy()
        listener_pred = np.full(len(y), -1)
        listener_pred[has_vote] = lab.encode(vote[has_vote])
        s["listeners"] = {
            "n_clips_with_majority": int(has_vote.sum()),
            # The listener majority as a "classifier" of the intended emotion: a human reference.
            "listener_majority_vs_intended": bs.interval(listener_pred, has_vote).as_dict(),
            "model_vs_intended_same_clips": bs.interval(p, has_vote).as_dict(),
            # How well the model agrees with what people heard.
            "model_vs_listener_majority": speaker_bootstrap(
                listener_pred[has_vote],
                p[has_vote],
                spk[has_vote],
                K,
                ev.n_bootstrap,
                cfg.seed,
                ev.ci_level,
            )[0].as_dict(),
        }

        plots.confusion_matrix(
            confusion(y, p, K),
            lab.emotions,
            run / f"confusion_{model}.png",
            f"{plots.LABELS[model]}: speaker-independent split",
        )
        summary[model] = s
        headline[model] = p
        log.info(
            "%s: UAR speaker-independent %s, random split %s",
            model,
            s["speaker"]["uar"],
            s["random"]["uar"],
        )

    if {"wavlm", "hubert"} <= set(headline):
        summary["wavlm_minus_hubert"] = bs.paired(headline["wavlm"], headline["hubert"]).as_dict()

    plots.layer_profile(
        profiles,
        lab.chance,
        run / "layer_profile.png",
        "UAR by layer (linear probe, 95% speaker-bootstrap interval)",
        random_profiles,
    )
    preds.to_parquet(run / "predictions.parquet", index=False)
    write_json(run / "summary.json", summary)
    log.info("done: %s", run)


if __name__ == "__main__":
    main()
