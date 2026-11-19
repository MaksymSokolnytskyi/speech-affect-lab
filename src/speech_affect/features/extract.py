"""Batch embedding extraction with an on-disk cache.

Output: `data/cache/embeddings/<model>/<condition>.npz` with fields
- `clip_id`: `[N]` strings in manifest order;
- `emb`: `[N, 13, 768]` float16 (half precision halves the size and is ample for a
  logistic regression);
- `meta`: JSON with the model, pinned HF revision, condition, timestamp and duration.

Usage:
    uv run python -m speech_affect.features.extract --model wavlm --condition clean
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speech_affect.channel.conditions import get_condition
from speech_affect.config import Config, get_config
from speech_affect.data.cremad import load_manifest
from speech_affect.features.encoders import load_encoder

log = logging.getLogger(__name__)


def embeddings_path(cfg: Config, model: str, condition: str) -> Path:
    return cfg.paths.cache / "embeddings" / model / f"{condition}.npz"


def load_wave(path: Path, expected_sr: int) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr}, expected {expected_sr}")
    if wave.ndim != 1:
        raise ValueError(f"{path}: expected mono audio")
    return wave


def extract(model: str, condition: str, overwrite: bool = False, limit: int | None = None):
    cfg = get_config()
    out = embeddings_path(cfg, model, condition)
    if out.exists() and not overwrite and limit is None:
        log.info("already exists: %s", out)
        return out
    manifest = load_manifest(cfg)
    if limit is not None:
        manifest = manifest.head(limit)
    transform = get_condition(condition)
    enc = load_encoder(model)
    sr = cfg.audio.sample_rate

    embs = np.empty(
        (len(manifest), cfg.encoders.n_hidden_states, cfg.encoders.hidden_size), np.float16
    )
    t0 = time.time()
    for i, row in enumerate(tqdm(manifest.itertuples(), total=len(manifest), desc=model)):
        wave = transform(load_wave(cfg.paths.root / row.path, sr), row.clip_id)
        embs[i] = enc.embed(wave, sr)
    meta = {
        "model": model,
        "hf_id": enc.hf_id,
        "revision": enc.revision,
        "condition": condition,
        "n_clips": len(manifest),
        "device": str(enc.device),
        "seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if limit is not None:
        log.info("dry run on %d clips: %s", limit, meta)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, clip_id=manifest["clip_id"].to_numpy().astype(str), emb=embs, meta=json.dumps(meta)
    )
    log.info("saved %s: %s", out, meta)
    return out


def load_embeddings(model: str, condition: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = embeddings_path(get_config(), model, condition)
    with np.load(path, allow_pickle=False) as z:
        return z["clip_id"], z["emb"], json.loads(str(z["meta"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Extract per-layer embeddings for one condition.")
    p.add_argument("--model", required=True, choices=sorted(get_config().encoders.models))
    p.add_argument("--condition", default="clean")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--limit", type=int, default=None, help="dry run on the first N clips (nothing is saved)"
    )
    a = p.parse_args()
    extract(a.model, a.condition, a.overwrite, a.limit)


if __name__ == "__main__":
    main()
