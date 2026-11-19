"""Render the report figures from saved run artifacts (no model is re-run).

Every figure is drawn from `outputs/<experiment>/<latest run>/` (summary.json, predictions)
and the data manifest, and written twice: vector PDF for the LaTeX report and PNG for the
README, both into `report/figures/`.

    uv run python experiments/make_figures.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speech_affect.config import get_config
from speech_affect.data.cremad import load_manifest
from speech_affect.evaluation.metrics import confusion
from speech_affect.evaluation.plots import BLUES, CATEGORICAL, GRID, INK, LABELS, MUTED, SERIES
from speech_affect.runs import latest_run

log = logging.getLogger("make_figures")
cfg = get_config()
FIG_DIR = cfg.paths.root / "report" / "figures"
EMOTION_NAMES = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
CONDITION_NAMES = {
    "clean": "clean",
    "tel": "phone",
    "tel_snr20": "phone\n+20 dB",
    "tel_snr10": "phone\n+10 dB",
    "tel_snr5": "phone\n+5 dB",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }
)


def _style(ax, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", name)


def _summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def _row_normalised_heatmap(ax, m: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(m, cmap=BLUES, vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                f"{m[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if m[i, j] > 0.55 else INK,
            )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title, loc="left", color=INK)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_listeners(man: pd.DataFrame) -> None:
    """Intended emotion vs. what the audio-only listener majority reported."""
    emo = list(cfg.cremad.emotions)
    man = man[man["voice_vote"].str.len() == 1]  # clips with a unique majority (no tie)
    m = pd.crosstab(man["emotion"], man["voice_vote_emotion"], normalize="index")
    m = m.reindex(index=emo, columns=emo, fill_value=0.0).to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    _row_normalised_heatmap(ax, m, [EMOTION_NAMES[e] for e in emo], "")
    ax.set_xlabel("listener majority (audio only)", color=INK)
    ax.set_ylabel("intended by the actor", color=INK)
    _save(fig, "listeners")


def fig_layers(r1: Path, r4: Path | None) -> None:
    s = _summary(r1)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for m in ("wavlm", "hubert"):
        p = s[m]["layer_profile_speaker"]
        x = np.arange(len(p["point"]))
        c = SERIES[m]
        ax.fill_between(x, p["lo"], p["hi"], color=c, alpha=0.15, linewidth=0)
        ax.plot(
            x,
            p["point"],
            color=c,
            lw=1.8,
            marker="o",
            ms=4,
            mec="white",
            mew=0.8,
            label=f"{LABELS[m]}, speaker-independent",
        )
        ax.plot(
            x,
            s[m]["layer_profile_random"],
            color=c,
            lw=1.2,
            ls="--",
            label=f"{LABELS[m]}, random split (leaky)",
        )
    if r4 is not None:
        e = _summary(r4)["egemaps"]["uar"]["point"]
        ax.axhline(e, color=MUTED, lw=1, ls=":")
        ax.text(
            12.3,
            e + 0.005,
            f"eGeMAPS baseline, 88 features ({e:.2f})",
            color=MUTED,
            fontsize=7.5,
            ha="right",
            va="bottom",
        )
    ax.set_xticks(range(13))
    ax.set_xlabel("hidden state (0 = transformer input, 12 = last layer)", color=INK)
    ax.set_ylabel("UAR (chance = 0.167)", color=INK)
    ax.set_ylim(0.50, 0.80)
    _style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _save(fig, "layers")


def fig_confusions(r1: Path, r4: Path, man: pd.DataFrame) -> None:
    emo = list(cfg.cremad.emotions)
    y = man["emotion"].map({e: i for i, e in enumerate(emo)}).to_numpy()
    p_w = pd.read_parquet(r1 / "predictions.parquet")["pred_wavlm_speaker"].to_numpy()
    p_e = pd.read_parquet(r4 / "predictions.parquet")["pred_egemaps_speaker"].to_numpy()
    names = [EMOTION_NAMES[e] for e in emo]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), layout="constrained")
    for ax, pred, title in ((axes[0], p_e, "(a) eGeMAPS"), (axes[1], p_w, "(b) WavLM Base+")):
        cm = confusion(y, pred, len(emo))
        im = _row_normalised_heatmap(ax, cm / cm.sum(1, keepdims=True), names, title)
        ax.set_xlabel("predicted", color=INK)
    axes[0].set_ylabel("intended", color=INK)
    axes[1].set_yticklabels([])
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    _save(fig, "confusions")


def fig_channel(r2: Path) -> None:
    s = _summary(r2)
    conds = s["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True, layout="constrained")
    for ax, m in zip(axes, ("wavlm", "hubert"), strict=True):
        for si, (key, name) in enumerate(
            (("A_train_clean", "trained on clean"), ("B_train_aug", "trained with augmentation"))
        ):
            x = np.arange(len(conds)) + (si - 0.5) * 0.22
            pt = np.array([s[m][key][c]["point"] for c in conds])
            lo = np.array([s[m][key][c]["lo"] for c in conds])
            hi = np.array([s[m][key][c]["hi"] for c in conds])
            ax.errorbar(
                x,
                pt,
                yerr=[pt - lo, hi - pt],
                fmt="none",
                ecolor=CATEGORICAL[si],
                elinewidth=1.6,
                alpha=0.8,
            )
            ax.plot(x, pt, "o", color=CATEGORICAL[si], ms=4.5, mec="white", mew=0.8, label=name)
        ax.axhline(s["chance"], color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(len(conds)), [CONDITION_NAMES[c] for c in conds])
        ax.set_title(LABELS[m], loc="left", color=INK)
        _style(ax)
    axes[0].set_ylabel("UAR", color=INK)
    axes[0].text(4.4, s["chance"] + 0.008, "chance", color=MUTED, fontsize=7, ha="right")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
    )
    _save(fig, "channel")


def fig_compression(r3: Path) -> None:
    s = _summary(r3)
    lat, k = s["latency"], s["layer"]
    variants = [
        ("torch_full_fp32", "PyTorch, 12 layers, fp32"),
        ("onnx_full_fp32", "ONNX, 12 layers, fp32"),
        ("onnx_full_int8", "ONNX, 12 layers, int8"),
        ("onnx_trunc_fp32", f"ONNX, {k} layers, fp32"),
        ("onnx_trunc_int8", f"ONNX, {k} layers, int8"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(len(variants))[::-1]
    for t, c, off in (("t1", CATEGORICAL[0], 0.19), ("t4", CATEGORICAL[1], -0.19)):
        vals = [lat[f"{v}_{t}"]["p50_ms"] for v, _ in variants]
        ax.barh(
            y + off, vals, height=0.36, color=c, label=f"{t[1]} thread{'s' if t != 't1' else ''}"
        )
        for yi, v in zip(y + off, vals, strict=True):
            ax.text(v + 8, yi, f"{v:.0f}", va="center", fontsize=7, color=INK)
    for yi, (v, _) in zip(y, variants, strict=True):
        size = lat[f"{v}_t1"].get("size_mb")
        if size:
            ax.text(905, yi, f"{size:.0f} MB", va="center", ha="right", fontsize=7, color=MUTED)
    ax.set_yticks(y, [name for _, name in variants])
    ax.set_xlim(0, 910)
    ax.set_xlabel("median latency for a 3 s clip, ms (CPU, ONNX Runtime)", color=INK)
    _style(ax, grid_axis="x")
    ax.text(905, len(variants) - 0.35, "model size", ha="right", fontsize=7, color=MUTED)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.25))
    _save(fig, "compression")


def fig_representations(r5: Path) -> None:
    s = _summary(r5)
    cka = np.array(s["cka"]["wavlm_x_hubert"])
    sh = s["emotion_subspace"]["split_half_speakers"]
    rnd = s["emotion_subspace"]["grassmann_random_pairs"]["mean"]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]}
    )
    im = axes[0].imshow(cka, cmap=BLUES, vmin=0, vmax=1, origin="lower")
    axes[0].set_xlabel("HuBERT layer", color=INK)
    axes[0].set_ylabel("WavLM layer", color=INK)
    axes[0].set_xticks(range(0, 13, 2))
    axes[0].set_yticks(range(0, 13, 2))
    axes[0].set_title("(a) linear CKA, WavLM × HuBERT", loc="left", color=INK)
    axes[0].tick_params(colors=MUTED, labelcolor=INK)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    x = np.arange(len(sh["across_aligned_on_a_compared_on_b"]))
    ax = axes[1]
    ax.plot(
        x,
        sh["across_aligned_on_a_compared_on_b"],
        color=CATEGORICAL[2],
        lw=1.8,
        marker="o",
        ms=4,
        mec="white",
        mew=0.8,
        label="WavLM vs HuBERT (rotation fit on half A)",
    )
    ax.plot(
        x,
        sh["noise_floor_wavlm"],
        color=SERIES["wavlm"],
        lw=1.2,
        ls="--",
        label="WavLM: half A vs half B",
    )
    ax.plot(
        x,
        sh["noise_floor_hubert"],
        color=SERIES["hubert"],
        lw=1.2,
        ls="--",
        label="HuBERT: half A vs half B",
    )
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("layer", color=INK)
    ax.set_ylabel("Grassmann distance", color=INK)
    ax.set_title(
        f"(b) emotion subspaces, held-out half B\n(random pair ≈ {rnd:.2f})", loc="left", color=INK
    )
    _style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _save(fig, "representations")


def fig_fisher(r1: Path, r5: Path) -> None:
    s5 = _summary(r5)
    if "fisher" not in s5:
        log.warning("exp05 run has no Fisher criteria; re-run exp05 to draw fig_fisher")
        return
    s1 = _summary(r1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), layout="constrained")
    for m in ("wavlm", "hubert"):
        f = s5["fisher"][m]
        x = np.arange(len(f["naive"]))
        axes[0].plot(x, f["naive"], color=SERIES[m], lw=1.6, marker="o", ms=3.5, label=LABELS[m])
        r = np.corrcoef(f["lda"], s1[m]["layer_profile_speaker"]["point"])[0, 1]
        axes[1].plot(
            x,
            f["lda"],
            color=SERIES[m],
            lw=1.6,
            marker="o",
            ms=3.5,
            label=f"{LABELS[m]} (r with UAR = {r:.2f})",
        )
    axes[0].set_title("(a) between/within ratio, Euclidean", loc="left", color=INK)
    axes[1].set_title(r"(b) Fisher–LDA $J=\mathrm{tr}(S_W^{-1}S_B)$", loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("layer", color=INK)
        ax.set_xticks(range(0, 13, 2))
        _style(ax)
        ax.legend(frameon=False, fontsize=7)
    _save(fig, "fisher")


def fig_variance(r5: Path) -> None:
    ve = _summary(r5)["variance_explained"]
    factors = (
        ("speaker", "speaker", "-"),
        ("sentence", "sentence", "--"),
        ("emotion", "emotion", ":"),
    )
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for m in ("wavlm", "hubert"):
        for key, name, ls in factors:
            ax.plot(
                range(len(ve[m][key])),
                ve[m][key],
                color=SERIES[m],
                ls=ls,
                lw=1.6,
                marker="o" if ls == "-" else None,
                ms=3,
                label=f"{LABELS[m]}: {name}",
            )
    ax.set_xticks(range(13))
    ax.set_xlabel("layer", color=INK)
    ax.set_ylabel(r"variance explained, $\eta^2$", color=INK)
    _style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    _save(fig, "variance")


def write_recall_tex(r4: Path) -> None:
    """Per-class recall for the 'emotion rose' TikZ figure, as LaTeX macros."""
    r = _summary(r4)["recall_per_class"]
    angles = (90, 30, -30, -90, -150, 150)
    lines = [
        "% Generated by experiments/make_figures.py from the exp04 summary.json -- do not edit.",
        "% \\petal{angle}{eGeMAPS recall}{WavLM recall}{label}",
        "\\newcommand{\\recallpetals}{%",
        *(
            f"\\petal{{{a}}}{{{e:.2f}}}{{{w:.2f}}}{{{EMOTION_NAMES[c]}}}%"
            for a, c, e, w in zip(angles, r["emotions"], r["egemaps"], r["wavlm"], strict=True)
        ),
        "}",
    ]
    out = cfg.paths.root / "report" / "generated" / "recall.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    man = load_manifest(cfg)
    runs = {
        e: latest_run(e)
        for e in (
            "exp01_emotion_probe",
            "exp02_telephone",
            "exp03_compression",
            "exp04_egemaps_baseline",
            "exp05_representations",
        )
    }
    fig_listeners(man)
    r1, r4, r5 = (
        runs["exp01_emotion_probe"],
        runs["exp04_egemaps_baseline"],
        runs["exp05_representations"],
    )
    if r1:
        fig_layers(r1, r4)
    if r1 and r4:
        fig_confusions(r1, r4, man)
    if runs["exp02_telephone"]:
        fig_channel(runs["exp02_telephone"])
    if runs["exp03_compression"]:
        fig_compression(runs["exp03_compression"])
    if r5:
        fig_representations(r5)
    if r1 and r5:
        fig_fisher(r1, r5)
    if r5:
        fig_variance(r5)
    if r4:
        write_recall_tex(r4)


if __name__ == "__main__":
    main()
