"""Build the README results block from the latest finished run of every experiment.

No number in the README is typed by hand: this script reads
`outputs/<experiment>/<timestamp>/summary.json` and renders Markdown tables that cite the run
they come from. Figures are referenced from `report/figures/` (rendered by
`experiments/make_figures.py`).

The block replaces whatever sits between the `<!-- RESULTS -->` and `<!-- /RESULTS -->`
markers in README.md and is also written to `outputs/summary/results.md`.

    uv run python experiments/make_report.py              # update README + results.md
    uv run python experiments/make_report.py --print-only # just print the block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_affect.config import get_config
from speech_affect.evaluation.plots import CONDITION_LABELS, LABELS
from speech_affect.runs import latest_run

cfg = get_config()
SUMMARY_DIR = cfg.paths.outputs / "summary"
FIGURES = "report/figures"
MARKERS = ("<!-- RESULTS -->", "<!-- /RESULTS -->")

# Pre-registered thresholds (docs/PREREGISTRATION.md).
INT8_MAX_DROP = 0.01
"""int8: lower bound of the paired UAR(int8) - UAR(fp32) interval must be >= -1 pp."""
INT8_MIN_SPEEDUP = 1.5
"""int8: p50 latency must drop by at least this factor."""


def load(run: Path, name: str = "summary.json") -> dict:
    return json.loads((run / name).read_text(encoding="utf-8"))


def ci(d: dict, pct: bool = False) -> str:
    k = 100 if pct else 1
    fmt = "{:+.1f}" if pct else "{:.3f}"
    return f"{fmt.format(d['point'] * k)} [{fmt.format(d['lo'] * k)}, {fmt.format(d['hi'] * k)}]"


def fig(name: str, alt: str) -> str:
    return f"![{alt}]({FIGURES}/{name}.png)"


def section_exp01(run: Path) -> list[str]:
    s = load(run)
    k = round(1 / s["chance"])
    lines = [
        f"### 1. Emotion on clean audio (CREMA-D, {k} classes, chance = {s['chance']:.3f})",
        "",
        f"Run: `outputs/exp01_emotion_probe/{run.name}`. UAR with a 95% speaker-bootstrap "
        "interval; differences by paired bootstrap, in percentage points (pp).",
        "",
        "| Encoder | UAR, speaker-independent | UAR, random split "
        "| Leakage (random - speaker), pp | Layer chosen per fold |",
        "|---|---|---|---|---|",
    ]
    for m in LABELS:
        if m not in s:
            continue
        x = s[m]
        layers = ", ".join(str(c["layer"]) for c in x["speaker"]["chosen"])
        lines.append(
            f"| {LABELS[m]} | {ci(x['speaker']['uar'])} | {ci(x['random']['uar'])} | "
            f"{ci(x['leak_random_minus_speaker'], pct=True)} | {layers} |"
        )
    if "wavlm_minus_hubert" in s:
        lines += ["", f"WavLM - HuBERT (paired): {ci(s['wavlm_minus_hubert'], pct=True)} pp."]
    w = s.get("wavlm")
    if w:
        lst = w["listeners"]
        lines += [
            "",
            "**Slices (WavLM, speaker-independent split):**",
            "",
            "| Slice | UAR |",
            "|---|---|",
            f"| female speakers | {ci(w['by_sex']['Female'])} |",
            f"| male speakers | {ci(w['by_sex']['Male'])} |",
            f"| female - male, pp | {ci(w['by_sex']['female_minus_male'], pct=True)} |",
            *[
                f"| intensity {lvl} (sentence IEO) | {ci(w['by_intensity_IEO'][lvl])} |"
                for lvl in ("LO", "MD", "HI")
            ],
            f"| listener majority vs intended emotion ({lst['n_clips_with_majority']} clips "
            f"without a tie) | {ci(lst['listener_majority_vs_intended'])} |",
            "| model vs intended emotion (same clips) | "
            f"{ci(lst['model_vs_intended_same_clips'])} |",
            f"| model vs listener majority | {ci(lst['model_vs_listener_majority'])} |",
        ]
    lines += ["", fig("layers", "UAR by layer"), ""]
    return lines


def section_exp02(run: Path) -> list[str]:
    s = load(run)
    lines = [
        "### 2. Telephone channel and noise",
        "",
        f"Run: `outputs/exp02_telephone/{run.name}`. A: probe trained on clean audio; "
        "B: trained on augmented audio (telephone and noise from a separate train pool of "
        "noise files).",
        "",
    ]
    for m in LABELS:
        if m not in s:
            continue
        x = s[m]
        lines += [
            f"**{LABELS[m]}**",
            "",
            "| Test condition | A: trained on clean | Drop in A (clean - condition), pp "
            "| B: with augmentation | Gain B - A, pp |",
            "|---|---|---|---|---|",
        ]
        for c in s["conditions"]:
            drop = ci(x["drop_clean_minus_cond_A"][c], pct=True) if c != "clean" else "-"
            lines.append(
                f"| {CONDITION_LABELS[c]} | {ci(x['A_train_clean'][c])} | {drop} | "
                f"{ci(x['B_train_aug'][c])} | {ci(x['gain_B_minus_A'][c], pct=True)} |"
            )
        best = ", ".join(
            f"{CONDITION_LABELS[c]}: {v['layer']}"
            for c, v in x["best_layer_by_condition_A"].items()
        )
        lines += ["", f"Best layer by condition (A): {best}.", ""]
    lines += [fig("channel", "UAR by test condition"), ""]
    return lines


def section_exp03(run: Path) -> list[str]:
    s = load(run)
    meta = load(run, "meta.json")
    bench = meta["config"]["export"]  # the settings this run actually used
    lat, k = s["latency"], s["layer"]
    n_full = meta.get("n_layers_full", cfg.encoders.n_layers)
    threads = [f"t{t}" for t in bench["bench_threads"]]

    def n_thr(t: str) -> str:
        n = int(t[1:])
        return f"{n} thread" + ("s" if n > 1 else "")

    lines = [
        "### 3. Compression for CPU",
        "",
        f"Run: `outputs/exp03_compression/{run.name}`. {LABELS[s['model']]} truncated to "
        f"{k} layers. {bench['bench_seconds']:g} s clip, ONNX Runtime on CPU, p50/p95 over "
        f"{bench['bench_runs']} runs after {bench['bench_warmup']} warm-up runs.",
        "",
        "| Variant | Size, MB | "
        + " | ".join(f"p50 / p95, ms ({n_thr(t)})" for t in threads)
        + " |",
        "|---|---|" + "---|" * len(threads),
    ]
    variants = [
        ("torch_full_fp32", f"PyTorch, {n_full} layers, fp32"),
        ("onnx_full_fp32", f"ONNX, {n_full} layers, fp32"),
        ("onnx_full_int8", f"ONNX, {n_full} layers, int8"),
        ("onnx_trunc_fp32", f"ONNX, {k} layers, fp32"),
        ("onnx_trunc_int8", f"ONNX, {k} layers, int8"),
    ]
    for v, label in variants:
        size = lat[f"{v}_{threads[0]}"].get("size_mb")
        cells = [
            f"{lat[f'{v}_{t}']['p50_ms']:.0f} / {lat[f'{v}_{t}']['p95_ms']:.0f}" for t in threads
        ]
        lines.append(
            f"| {label} | {'-' if size is None else f'{size:.0f}'} | {' | '.join(cells)} |"
        )
    sp, eq = s["speedup_p50"], s["equivalence"]

    def per_thread(key: str) -> str:
        return ", ".join(f"{sp[t][key]:.2f}x ({n_thr(t)})" for t in threads)

    lines += [
        "",
        f"- int8 vs fp32 speed-up (truncated model, p50): {per_thread('int8_vs_fp32_trunc')}.",
        f"- {k} vs {n_full} layers (fp32, p50): {per_thread('trunc_vs_full_fp32')}.",
        f"- Combined (PyTorch {n_full} layers fp32 -> ONNX {k} layers int8): "
        f"{per_thread('trunc_int8_vs_full_fp32_torch')}.",
        f"- ONNX fp32 vs PyTorch equivalence: max abs difference "
        f"{eq['full']['max_abs_diff']:.1e} ({n_full} layers), {eq['trunc']['max_abs_diff']:.1e} "
        f"({k} layers) on {eq['trunc']['n_clips']} clips; threshold "
        f"{cfg.export.equivalence_atol:.0e}.",
    ]
    q = s.get("quality")
    if q:
        lines.append(
            f"- Quality on all clips (CPU, out-of-fold): fp32 {ci(q['uar_trunc_fp32'])}, "
            f"int8 {ci(q['uar_trunc_int8'])}; int8 - fp32 "
            f"{ci(q['int8_minus_fp32'], pct=True)} pp; predictions agree on "
            f"{q['prediction_agreement'] * 100:.1f}% of clips; fp32/int8 embedding cosine "
            f"mean {q['embedding_cosine']['mean']:.3f}, min {q['embedding_cosine']['min']:.3f}."
        )
    lines += [f"- CPU: {meta['cpu']}.", ""]
    return lines


def section_profile(run: Path) -> list[str]:
    s = load(run)
    lines = [
        "**Where the CPU time goes** (exploratory, not pre-registered; ONNX Runtime profiler, "
        f"1 thread, `outputs/exp03b_profile/{run.name}`):",
        "",
        "| Model | Share of time by operator type |",
        "|---|---|",
    ]
    for name, r in s.items():
        top = ", ".join(f"{k} {v:.0%}" for k, v in list(r["share_by_op"].items())[:4])
        lines.append(f"| {name} | {top} |")
    return [*lines, ""]


def section_exp04(run: Path) -> list[str]:
    s = load(run)
    r = s["recall_per_class"]
    n = s["n_features"]
    return [
        "### 4. Baseline: classic eGeMAPS features",
        "",
        f"Run: `outputs/exp04_egemaps_baseline/{run.name}`. {n} openSMILE features, the same "
        "logistic regression, the same nested speaker-independent validation, the same folds.",
        "",
        "| Features | UAR, speaker-independent |",
        "|---|---|",
        f"| eGeMAPS ({n}) | {ci(s['egemaps']['uar'])} |",
        f"| {LABELS['hubert']}, linear probe | {ci(s['hubert']['uar'])} |",
        f"| {LABELS['wavlm']}, linear probe | {ci(s['wavlm']['uar'])} |",
        "",
        f"WavLM - eGeMAPS (paired): {ci(s['wavlm_minus_egemaps'], pct=True)} pp. "
        "Per-class recall (eGeMAPS -> WavLM): "
        + ", ".join(
            f"{e} {a:.2f} -> {b:.2f}"
            for e, a, b in zip(r["emotions"], r["egemaps"], r["wavlm"], strict=True)
        )
        + ".",
        "",
        fig("confusions", "Confusion matrices, eGeMAPS vs WavLM"),
        "",
    ]


def section_exp05(run: Path) -> list[str]:
    s = load(run)
    es = s["emotion_subspace"]
    sh = es["split_half_speakers"]
    cka = s["cka"]["wavlm_x_hubert"]
    diag = [cka[i][i] for i in range(len(cka))]
    lo, hi = 4, 9  # the middle layers, where the probe performs best
    mid = slice(lo, hi + 1)
    across = sh["across_aligned_on_a_compared_on_b"][mid]
    floor = sh["noise_floor_wavlm"][mid]
    return [
        "### 5. Exploratory: WavLM vs HuBERT representations",
        "",
        f"Run: `outputs/exp05_representations/{run.name}`. No pre-registered criteria.",
        "",
        f"- Linear CKA between corresponding layers: {min(diag):.2f}-{max(diag):.2f}.",
        f"- Grassmann distance between the {es['dim']}-dimensional emotion class-mean "
        f"subspaces, layers {lo}-{hi} (rotation fit on one half of the actors, compared on the "
        f"other): WavLM vs HuBERT {min(across):.2f}-{max(across):.2f}; the same model on the "
        f"two halves {min(floor):.2f}-{max(floor):.2f}; random subspaces "
        f"{es['grassmann_random_pairs']['mean']:.2f}.",
        "",
        fig("representations", "Representation comparison"),
        "",
    ]


def _verdict(ok: bool) -> str:
    return "met" if ok else "**not met**"


def section_criteria() -> list[str]:
    """Verdicts against the pre-registered criteria (docs/PREREGISTRATION.md)."""
    rows = []
    if r1 := latest_run("exp01_emotion_probe"):
        s = load(r1)
        w = s["wavlm"]
        k = round(1 / s["chance"])
        rows.append(
            (
                "1",
                f"Speaker-independent UAR: lower bound of the 95% interval > chance (1/{k})",
                f"{w['speaker']['uar']['lo']:.3f} > {s['chance']:.3f}",
                _verdict(w["speaker"]["uar"]["lo"] > s["chance"]),
            )
        )
        leak = w["leak_random_minus_speaker"]
        rows.append(
            (
                "1",
                "Leakage: lower bound of the paired interval (random - speaker) > 0",
                f"{leak['lo'] * 100:+.1f} pp",
                _verdict(leak["lo"] > 0),
            )
        )
    if r2 := latest_run("exp02_telephone"):
        s = load(r2)
        w = s["wavlm"]
        tel = [c for c in s["conditions"] if c != "clean"]
        worst = min(w["gain_B_minus_A"][c]["lo"] for c in tel)
        rows.append(
            (
                "2",
                "Drop under telephone conditions reported with a paired interval",
                "see the stage 2 table",
                "met",
            )
        )
        rows.append(
            (
                "2",
                "Augmentation: lower bound of the paired B - A interval > 0 "
                "on every telephone condition",
                f"min lower bound {worst * 100:+.1f} pp",
                _verdict(worst > 0),
            )
        )
    if r3 := latest_run("exp03_compression"):
        s = load(r3)
        atol = cfg.export.equivalence_atol
        eq = max(v["max_abs_diff"] for v in s["equivalence"].values())
        rows.append(
            (
                "3",
                f"ONNX = PyTorch: max abs difference <= {atol:.0e}",
                f"{eq:.1e}",
                _verdict(eq <= atol),
            )
        )
        if q := s.get("quality"):
            lo = q["int8_minus_fp32"]["lo"]
            rows.append(
                (
                    "3",
                    "int8: lower bound of the paired interval UAR(int8) - UAR(fp32) "
                    f">= -{INT8_MAX_DROP * 100:g} pp",
                    f"{lo * 100:+.2f} pp",
                    _verdict(lo >= -INT8_MAX_DROP),
                )
            )
        sp = min(v["int8_vs_fp32_trunc"] for v in s["speedup_p50"].values())
        rows.append(
            (
                "3",
                f"int8: p50 latency lower by at least {INT8_MIN_SPEEDUP:g}x",
                f"{sp:.2f}x",
                _verdict(sp >= INT8_MIN_SPEEDUP),
            )
        )
    if r4 := latest_run("exp04_egemaps_baseline"):
        d = load(r4)["wavlm_minus_egemaps"]
        rows.append(
            (
                "extra",
                "WavLM beats eGeMAPS: lower bound of the paired difference interval > 0",
                f"{d['lo'] * 100:+.1f} pp",
                _verdict(d["lo"] > 0),
            )
        )
    return [
        "### Success criteria (pre-registered in docs/PREREGISTRATION.md)",
        "",
        "| Stage | Criterion | Value | Verdict |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows],
        "",
    ]


def build_block() -> str:
    parts = section_criteria()
    for exp, fn in (
        ("exp01_emotion_probe", section_exp01),
        ("exp02_telephone", section_exp02),
        ("exp03_compression", section_exp03),
        ("exp03b_profile", section_profile),
        ("exp04_egemaps_baseline", section_exp04),
        ("exp05_representations", section_exp05),
    ):
        if (run := latest_run(exp)) is not None:
            parts += fn(run)
    return "\n".join(parts).strip()


def write_readme(block: str, readme: Path) -> None:
    start, end = MARKERS
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise RuntimeError(f"{readme} has no {start} ... {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the README results block.")
    ap.add_argument(
        "--print-only", action="store_true", help="print the block without writing any file"
    )
    args = ap.parse_args()
    block = build_block()
    if not args.print_only:
        write_readme(block, cfg.paths.root / "README.md")
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        (SUMMARY_DIR / "results.md").write_text(block + "\n", encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
