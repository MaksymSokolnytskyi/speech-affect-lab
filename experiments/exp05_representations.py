"""Exploratory (no pre-registered criteria): how WavLM and HuBERT representations are
organised across layers.

1. Layer-by-layer linear CKA: WavLM x WavLM, HuBERT x HuBERT, WavLM x HuBERT (13 x 13 each).
2. Procrustes distance between WavLM layer k and HuBERT layer k.
3. Variance decomposition: share of a layer's embedding variance explained by speaker,
   emotion, sentence and sex.
4. "Emotion subspaces" (spanned by the class means, dimension 5) and Grassmann distances
   between them: across layers of one model, and across models after Procrustes alignment,
   with a split-half speaker control.
5. Class separability per layer: Euclidean between/within ratio vs. Fisher-LDA criterion.

Clean embeddings of all 7,442 clips. Labels are used only in items 3-5.

    uv run python experiments/exp05_representations.py
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import orthogonal_procrustes

from speech_affect.analysis.representations import (
    cka_matrix,
    class_mean_subspace,
    fisher_lda,
    fisher_ratio_euclidean,
    grassmann_distance,
    linear_cka,
    procrustes_distance,
    standardize,
    variance_explained,
)
from speech_affect.config import get_config
from speech_affect.experiment import load_aligned_embeddings, load_labels, setup_logging
from speech_affect.runs import new_run, write_json

log = logging.getLogger("exp05")
MODELS = ("wavlm", "hubert")


def main() -> None:
    setup_logging()
    cfg = get_config()
    run = new_run("exp05_representations")
    man = load_labels(cfg).manifest
    emb = {m: load_aligned_embeddings(m, "clean", man)[0].astype(np.float32) for m in MODELS}
    L = emb["wavlm"].shape[1]
    y = man["emotion"].to_numpy()

    # 1. CKA between every pair of layers.
    layers = {m: [emb[m][:, i] for i in range(L)] for m in MODELS}
    cka = {
        f"{a}_x_{b}": cka_matrix(layers[a], layers[b]).round(4).tolist()
        for a, b in (("wavlm", "wavlm"), ("hubert", "hubert"), ("wavlm", "hubert"))
    }
    # Sanity check of the float32 batched version against the float64 reference.
    ref = linear_cka(emb["wavlm"][:, 3], emb["hubert"][:, 5])
    if abs(cka["wavlm_x_hubert"][3][5] - ref) >= 1e-3:
        raise RuntimeError(f"cka_matrix disagrees with linear_cka: {cka['wavlm_x_hubert'][3][5]}")
    log.info("CKA done")

    # 2. Procrustes on the diagonal.
    procrustes = [procrustes_distance(emb["wavlm"][:, i], emb["hubert"][:, i]) for i in range(L)]

    # 3. Variance decomposition.
    factors = {
        "speaker": man["speaker_id"].to_numpy(),
        "emotion": y,
        "sentence": man["sentence"].to_numpy(),
        "sex": man["sex"].to_numpy(),
    }
    var_expl = {
        m: {f: [variance_explained(emb[m][:, i], g) for i in range(L)] for f, g in factors.items()}
        for m in MODELS
    }
    log.info("variance decomposition done")

    # 4. Emotion subspaces.
    sub = {m: [class_mean_subspace(emb[m][:, i], y) for i in range(L)] for m in MODELS}
    k = sub["wavlm"][0].shape[1]
    within = {
        m: [[grassmann_distance(sub[m][i], sub[m][j]) for j in range(L)] for i in range(L)]
        for m in MODELS
    }
    # Across models: first rotate standardised HuBERT onto WavLM (no labels involved).
    across = []
    for i in range(L):
        zw, zh = standardize(emb["wavlm"][:, i]), standardize(emb["hubert"][:, i])
        R, _ = orthogonal_procrustes(zh, zw)
        across.append(grassmann_distance(sub["wavlm"][i], R.T @ sub["hubert"][i]))
    rng = np.random.default_rng(cfg.seed)
    d = emb["wavlm"].shape[2]
    random_ref = [
        grassmann_distance(
            np.linalg.qr(rng.normal(size=(d, k)))[0], np.linalg.qr(rng.normal(size=(d, k)))[0]
        )
        for _ in range(100)
    ]

    # Control on disjoint speakers. Half A: fit the rotation only (no labels).
    # Half B: emotion subspaces. "Noise floor": the same model, half A vs half B.
    spk = man["speaker_id"].to_numpy()
    uniq = np.unique(spk)
    half_a = np.isin(spk, rng.permutation(uniq)[: len(uniq) // 2])
    half_b = ~half_a
    split_half = {
        "across_aligned_on_a_compared_on_b": [],
        "noise_floor_wavlm": [],
        "noise_floor_hubert": [],
    }
    for i in range(L):
        zw, zh = standardize(emb["wavlm"][:, i]), standardize(emb["hubert"][:, i])
        R, _ = orthogonal_procrustes(zh[half_a], zw[half_a])
        sw_b = class_mean_subspace(emb["wavlm"][half_b, i], y[half_b])
        sh_b = class_mean_subspace(emb["hubert"][half_b, i], y[half_b])
        split_half["across_aligned_on_a_compared_on_b"].append(grassmann_distance(sw_b, R.T @ sh_b))
        for m, s_b in (("wavlm", sw_b), ("hubert", sh_b)):
            s_a = class_mean_subspace(emb[m][half_a, i], y[half_a])
            split_half[f"noise_floor_{m}"].append(grassmann_distance(s_a, s_b))

    # 5. Separability: Euclidean ratio vs Fisher-LDA (Mahalanobis geometry).
    fisher = {
        m: {
            "naive": [fisher_ratio_euclidean(emb[m][:, i], y) for i in range(L)],
            "lda": [fisher_lda(emb[m][:, i], y) for i in range(L)],
        }
        for m in MODELS
    }
    log.info("Fisher criteria done")

    summary = {
        "n_clips": len(y),
        "cka": cka,
        "procrustes_wavlm_vs_hubert_same_layer": procrustes,
        "variance_explained": var_expl,
        "emotion_subspace": {
            "dim": k,
            "grassmann_within": within,
            "grassmann_wavlm_vs_hubert_aligned": across,
            "split_half_speakers": split_half,
            "grassmann_random_pairs": {
                "mean": float(np.mean(random_ref)),
                "min": float(np.min(random_ref)),
                "max_possible": float(np.pi / 2 * np.sqrt(k)),
            },
        },
        "fisher": fisher,
    }
    write_json(run / "summary.json", summary)
    log.info("saved %s", run / "summary.json")


if __name__ == "__main__":
    main()
