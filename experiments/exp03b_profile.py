"""Stage 3 (exploratory, not pre-registered): where does CPU time go in the ONNX models?

The ONNX Runtime profiler records the time of every graph node; we group it by operator type.
Question: is the int8 speed-up small because the CNN convolutions stay float32?

    uv run python experiments/exp03b_profile.py
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
from pathlib import Path

from speech_affect.config import get_config
from speech_affect.experiment import setup_logging
from speech_affect.export.benchmark import ort_session, random_waveform
from speech_affect.export.onnx_export import INPUT_NAME
from speech_affect.runs import latest_run, new_run, write_json

log = logging.getLogger("exp03b")


def profile(path: Path, threads: int, seconds: float, warmup: int, runs: int, prefix: str) -> dict:
    sess = ort_session(path, threads, profile_prefix=prefix)
    x = random_waveform(seconds, get_config().audio.sample_rate)
    for _ in range(warmup + runs):
        sess.run(None, {INPUT_NAME: x})
    trace = sess.end_profiling()
    events = json.loads(Path(trace).read_text(encoding="utf-8"))
    os.remove(trace)
    by_op: collections.Counter = collections.Counter()
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            by_op[e["args"].get("op_name", "?")] += e["dur"]
    total = sum(by_op.values())
    return {
        "ms_per_run_incl_warmup": total / (warmup + runs) / 1000,
        "share_by_op": {k: v / total for k, v in by_op.most_common()},
    }


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Profile the exp03 ONNX models by operator type.")
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="directory with the exp03 ONNX models (default: those of the latest exp03 run)",
    )
    args = ap.parse_args()
    cfg = get_config()
    models_dir = args.models_dir
    if models_dir is None:
        exp03 = latest_run("exp03_compression")
        if exp03 is None:
            raise FileNotFoundError("no finished exp03 run; pass --models-dir")
        models_dir = cfg.paths.data / "models" / exp03.name
    if not any(models_dir.glob("*.onnx")):
        raise FileNotFoundError(f"no .onnx files in {models_dir}")
    run = new_run("exp03b_profile", {"models_dir": str(models_dir)})
    result = {}
    for path in sorted(models_dir.glob("*.onnx")):
        result[path.stem] = profile(
            path,
            threads=1,
            seconds=cfg.export.bench_seconds,
            warmup=5,
            runs=20,
            prefix=str(run / path.stem),
        )
        top = ", ".join(
            f"{k} {v:.0%}" for k, v in list(result[path.stem]["share_by_op"].items())[:5]
        )
        log.info("%s: %s", path.stem, top)
    write_json(run / "summary.json", result)


if __name__ == "__main__":
    main()
