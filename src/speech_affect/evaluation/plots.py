"""Per-run diagnostic plots (matplotlib) and the shared display names.

Palette: categorical slots 1-2 (blue, orange) from a checked palette; confusion matrices use
a single-hue sequential scale (blue). Grid and axes are muted, lines 2 px.

`LABELS` (model display names) and `CONDITION_LABELS` / `CONDITION_TICKS` (test conditions)
are the single source of truth for experiment scripts and the README report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from speech_affect.config import get_config

SERIES = {"wavlm": "#2a78d6", "hubert": "#eb6834"}
LABELS = {"wavlm": "WavLM Base+", "hubert": "HuBERT Base"}
CONDITION_LABELS = {
    "clean": "clean",
    "tel": "telephone",
    **{f"tel_snr{s}": f"telephone + noise at {s} dB SNR" for s in get_config().noise.snr_db},
}
"""Test condition -> human-readable name (tables)."""
CONDITION_TICKS = {
    "clean": "clean",
    "tel": "tel.",
    **{f"tel_snr{s}": f"tel. +{s} dB" for s in get_config().noise.snr_db},
}
"""Test condition -> short name (plot ticks)."""
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dc"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def layer_profile(
    profiles: dict[str, dict[str, np.ndarray]],
    chance: float,
    out: Path,
    title: str,
    random_profiles: dict[str, np.ndarray] | None = None,
) -> None:
    """profiles[model] = {"point": [L], "lo": [L], "hi": [L]} for the speaker-independent split."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for model, p in profiles.items():
        x = np.arange(len(p["point"]))
        c = SERIES[model]
        ax.fill_between(x, p["lo"], p["hi"], color=c, alpha=0.15, linewidth=0)
        ax.plot(
            x,
            p["point"],
            color=c,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=1,
            label=f"{LABELS[model]}, speaker-independent",
        )
        best = int(np.argmax(p["point"]))
        ax.annotate(
            f"{p['point'][best]:.3f}",
            (best, p["point"][best]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=INK,
            fontsize=8,
        )
        if random_profiles and model in random_profiles:
            ax.plot(
                x,
                random_profiles[model],
                color=c,
                linewidth=1.5,
                linestyle="--",
                label=f"{LABELS[model]}, random split",
            )
    # The y axis is zoomed to the data; chance level goes into the label (it is far below).
    lows = [min(p["lo"]) for p in profiles.values()]
    highs = [max(p["hi"]) for p in profiles.values()]
    if random_profiles:
        highs += [max(r) for r in random_profiles.values()]
    ax.set_ylim(np.floor(min(lows) * 20) / 20, np.ceil(max(highs) * 20) / 20 + 0.02)
    ax.set_xticks(range(len(next(iter(profiles.values()))["point"])))
    ax.set_xlabel(f"layer (0 = transformer input); chance = {chance:.3f}", color=INK)
    ax.set_ylabel("UAR", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def condition_dots(
    series: dict[str, dict[str, list[float]]],
    conditions: list[str],
    chance: float,
    out: Path,
    title: str,
) -> None:
    """Point + 95% interval per condition; series[name] = {"point", "lo", "hi"} by condition.

    Colours are categorical slots in series order (never by rank).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    n = len(series)
    width = 0.18
    for si, (name, s) in enumerate(series.items()):
        c = CATEGORICAL[si]
        x = np.arange(len(conditions)) + (si - (n - 1) / 2) * width
        pt, lo, hi = (np.asarray(s[k]) for k in ("point", "lo", "hi"))
        ax.errorbar(
            x, pt, yerr=[pt - lo, hi - pt], fmt="none", ecolor=c, elinewidth=2, capsize=0, alpha=0.6
        )
        ax.plot(
            x,
            pt,
            "o",
            color=c,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            label=name,
            linestyle="none",
        )
        for xi, v in zip(x, pt, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (xi, v),
                textcoords="offset points",
                xytext=(9, -3),
                fontsize=7,
                color=INK,
            )
    ax.axhline(chance, color=MUTED, linewidth=1, linestyle=":")
    ax.text(-0.45, chance, "chance", va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_ylabel("UAR", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def confusion_matrix(cm: np.ndarray, labels: list[str], out: Path, title: str) -> None:
    """Row-normalised (per-class recall on the diagonal)."""
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = norm[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.55 else INK,
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted", color=INK)
    ax.set_ylabel("intended emotion", color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
