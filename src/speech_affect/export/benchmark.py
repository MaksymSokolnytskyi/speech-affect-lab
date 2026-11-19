"""CPU latency: p50/p95 per clip of a fixed length.

Warm-up runs (the first calls are slower: memory allocation, graph optimisation) are not
counted. The thread count is set explicitly because it determines both the latency of a single
request and how many requests a server can handle in parallel.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def ort_session(
    path: Path, threads: int, profile_prefix: str | None = None
) -> ort.InferenceSession:
    """CPU session with a fixed thread count; `profile_prefix` turns on the ORT profiler."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def random_waveform(seconds: float, sr: int) -> np.ndarray:
    """`[1, T]` float32 Gaussian noise: a fixed, reproducible benchmark input."""
    return np.random.default_rng(0).normal(0, 0.1, (1, int(seconds * sr))).astype(np.float32)


def _stats(times: list[float]) -> dict:
    t = np.array(times) * 1000
    return {
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ms": float(t.mean()),
        "n": len(t),
    }


def bench_ort(path: Path, threads: int, seconds: float, warmup: int, runs: int, sr: int) -> dict:
    sess = ort_session(path, threads)
    x = random_waveform(seconds, sr)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append(time.perf_counter() - t0)
    return _stats(times)


@torch.inference_mode()
def bench_torch(
    module: torch.nn.Module, threads: int, seconds: float, warmup: int, runs: int, sr: int
) -> dict:
    torch.set_num_threads(threads)
    x = torch.from_numpy(random_waveform(seconds, sr))
    module = module.cpu().eval()
    for _ in range(warmup):
        module(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        module(x)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def file_size_mb(path: Path) -> float:
    total = path.stat().st_size
    data = path.with_suffix(path.suffix + ".data")  # external weights, if the exporter wrote any
    if data.exists():
        total += data.stat().st_size
    return total / 2**20
