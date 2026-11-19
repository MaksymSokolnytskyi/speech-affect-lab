"""Stage 3. What does compression for CPU cost, and what does it buy?

Encoder variants (the output is the embedding fed to the classifier):
- full_fp32:  all transformer layers, float32 (ONNX and, for reference, PyTorch);
- full_int8:  all layers, dynamic int8 quantisation;
- trunc_fp32: only the first k layers (k = the layer chosen in stage 1), float32;
- trunc_int8: the same, int8.

What is measured:
1. Equivalence of ONNX fp32 and PyTorch on real clips of different lengths (criterion 1e-4).
2. Latency p50/p95 for a 3 s clip, 1 and 4 CPU threads; file size.
3. Quality: every clip through trunc_fp32 and trunc_int8 on CPU; the out-of-fold probe
   trained on fp32 embeddings is applied to int8 embeddings (as in a real deployment).
   Paired UAR difference and the share of clips whose predictions agree.

`summary.json` is written only once the run has finished, so a crashed run is never picked
up as "the latest run"; partial results go to `summary.partial.json` in the meantime.

    uv run python experiments/exp03_compression.py --model wavlm --layer 6
Without --layer, the layer chosen most often by the nested loop in the latest exp01 run is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from speech_affect.config import get_config
from speech_affect.data.splits import speaker_folds
from speech_affect.experiment import load_labels, make_bootstrappers, probe_kwargs, setup_logging
from speech_affect.export.benchmark import bench_ort, bench_torch, file_size_mb, ort_session
from speech_affect.export.onnx_export import INPUT_NAME, build, export, torch_embedding
from speech_affect.export.quantize import quantize
from speech_affect.features.extract import load_wave
from speech_affect.models.probe import FIT, nested_layer_probe
from speech_affect.runs import latest_run, new_run, write_json

log = logging.getLogger("exp03")


def layer_from_exp01(model: str) -> int:
    run = latest_run("exp01_emotion_probe")
    if run is None:
        raise FileNotFoundError("no finished exp01 run; pass --layer explicitly")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    chosen = summary[model]["speaker"]["chosen"]
    layer, _ = Counter(c["layer"] for c in chosen).most_common(1)[0]
    log.info(
        "layer from %s: %d (per-fold choices: %s)", run.name, layer, [c["layer"] for c in chosen]
    )
    return int(layer)


def embed_all(sess, man: pd.DataFrame, root: Path, sr: int, dim: int) -> np.ndarray:
    name = sess.get_inputs()[0].name
    out = np.empty((len(man), dim), np.float32)
    for i, row in enumerate(tqdm(man.itertuples(), total=len(man), desc="CPU embeddings")):
        out[i] = sess.run(None, {name: load_wave(root / row.path, sr)[None]})[0][0]
    return out


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Stage 3: ONNX export, int8, CPU latency.")
    ap.add_argument("--model", default="wavlm", choices=sorted(get_config().encoders.models))
    ap.add_argument("--layer", type=int, default=None, help="truncate after this many layers")
    ap.add_argument("--skip-quality", action="store_true", help="export and latency only")
    args = ap.parse_args()

    cfg = get_config()
    ex = cfg.export
    sr = cfg.audio.sample_rate
    layer = layer_from_exp01(args.model) if args.layer is None else args.layer
    if not 1 <= layer <= cfg.encoders.n_layers:
        ap.error(
            f"layer must be in 1..{cfg.encoders.n_layers} (got {layer}): hidden state 0 is the "
            "transformer input, so a model truncated to 0 layers has no transformer to export"
        )

    # 1. Build, export and quantise.
    modules = {"full": build(args.model), "trunc": build(args.model, layer)}
    run = new_run(
        "exp03_compression",
        {
            "model": args.model,
            "layer": layer,
            "n_layers_full": modules["full"].n_layers,
            "cpu": platform.processor(),
            "torch_threads_default": torch.get_num_threads(),
        },
    )
    log.info("run directory: %s", run)
    paths = {}
    for name, mod in modules.items():
        paths[f"{name}_fp32"] = export(mod, run / f"{args.model}_{name}_fp32.onnx", sr)
        paths[f"{name}_int8"] = quantize(
            paths[f"{name}_fp32"], run / f"{args.model}_{name}_int8.onnx"
        )

    # 2. Equivalence of ONNX fp32 and PyTorch on real clips spanning the length range.
    lab = load_labels(cfg)
    man = lab.manifest
    idx = np.linspace(0, len(man) - 1, ex.n_equivalence_clips).astype(int)
    by_len = man.sort_values("duration_s").iloc[idx]
    equivalence = {}
    for name, mod in modules.items():
        sess = ort_session(paths[f"{name}_fp32"], threads=4)
        diffs = []
        for row in by_len.itertuples():
            w = load_wave(cfg.paths.root / row.path, sr)
            onnx_out = sess.run(None, {INPUT_NAME: w[None]})[0]
            diffs.append(float(np.abs(onnx_out - torch_embedding(mod, w)).max()))
        equivalence[name] = {
            "max_abs_diff": max(diffs),
            "passes": max(diffs) <= ex.equivalence_atol,
            "n_clips": len(diffs),
        }
        log.info("equivalence %s: max abs difference %.2e", name, max(diffs))

    # 3. Latency.
    latency = {}
    bench = (ex.bench_seconds, ex.bench_warmup, ex.bench_runs, sr)
    for threads in ex.bench_threads:
        for variant, path in paths.items():
            latency[f"onnx_{variant}_t{threads}"] = {
                **bench_ort(path, threads, *bench),
                "size_mb": file_size_mb(path),
            }
            log.info("%s t%d: %s", variant, threads, latency[f"onnx_{variant}_t{threads}"])
        latency[f"torch_full_fp32_t{threads}"] = bench_torch(modules["full"], threads, *bench)
    speedups = {
        f"t{t}": {
            "int8_vs_fp32_trunc": latency[f"onnx_trunc_fp32_t{t}"]["p50_ms"]
            / latency[f"onnx_trunc_int8_t{t}"]["p50_ms"],
            "int8_vs_fp32_full": latency[f"onnx_full_fp32_t{t}"]["p50_ms"]
            / latency[f"onnx_full_int8_t{t}"]["p50_ms"],
            "trunc_vs_full_fp32": latency[f"onnx_full_fp32_t{t}"]["p50_ms"]
            / latency[f"onnx_trunc_fp32_t{t}"]["p50_ms"],
            "trunc_int8_vs_full_fp32_torch": latency[f"torch_full_fp32_t{t}"]["p50_ms"]
            / latency[f"onnx_trunc_int8_t{t}"]["p50_ms"],
        }
        for t in ex.bench_threads
    }
    summary = {
        "model": args.model,
        "layer": layer,
        "equivalence": equivalence,
        "latency": latency,
        "speedup_p50": speedups,
    }
    partial = run / "summary.partial.json"
    write_json(partial, summary)

    # 4. Quality of fp32 vs int8 (the truncated model is the deployment candidate).
    if not args.skip_quality:
        y, spk = lab.y, lab.speakers
        folds = speaker_folds(spk, lab.sex, cfg.eval.n_outer_folds, cfg.seed)
        dim = modules["trunc"].hidden_size
        E = {
            v: embed_all(
                ort_session(paths[f"trunc_{v}"], ex.embed_threads), man, cfg.paths.root, sr, dim
            )
            for v in ("fp32", "int8")
        }
        np.savez_compressed(run / "cpu_embeddings.npz", **E)
        res = nested_layer_probe(
            E["fp32"][:, None, :],
            y,
            spk,
            folds,
            X_eval={"int8": E["int8"][:, None, :]},
            **probe_kwargs(cfg),
        )
        p32, p8 = res.pred[FIT], res.pred["int8"]
        bs = make_bootstrappers(cfg, y, spk)

        cos = np.sum(E["fp32"] * E["int8"], 1) / (
            np.linalg.norm(E["fp32"], axis=1) * np.linalg.norm(E["int8"], axis=1)
        )
        summary["quality"] = {
            "uar_trunc_fp32": bs.interval(p32).as_dict(),
            "uar_trunc_int8": bs.interval(p8).as_dict(),
            "int8_minus_fp32": bs.paired(p8, p32).as_dict(),
            "prediction_agreement": float((p32 == p8).mean()),
            "embedding_cosine": {"mean": float(cos.mean()), "min": float(cos.min())},
            "chosen_C": res.chosen,
        }
        log.info(
            "quality: fp32 %s, int8 %s, agreement %.3f",
            bs.interval(p32),
            bs.interval(p8),
            summary["quality"]["prediction_agreement"],
        )

    for p in run.glob("*.onnx"):  # the models are large: keep them under data/, not outputs/
        target = cfg.paths.data / "models" / run.name / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        p.replace(target)
    write_json(run / "summary.json", summary)
    partial.unlink()
    log.info("done: %s", run)


if __name__ == "__main__":
    main()
