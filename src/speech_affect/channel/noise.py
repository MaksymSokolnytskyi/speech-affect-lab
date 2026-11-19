"""Noise: MUSAN/noise download, train/test split of the noise files, mixing at a given SNR.

Why the noise files are split in two: if the same noise recording appeared both in training
augmentation and in the test conditions, the model could learn to ignore that particular
recording, a quiet form of leakage.

Download:
    uv run python -m speech_affect.channel.noise download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from speech_affect.config import get_config
from speech_affect.io import DownloadJob, download_many

log = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
POOLS = ("train", "test")


def _list_tree(repo: str, path: str, timeout: float) -> list[dict]:
    with urllib.request.urlopen(HF_API.format(repo=repo, path=path), timeout=timeout) as r:
        return json.load(r)


def noise_root() -> Path:
    return get_config().paths.raw / "musan"


def download() -> Path:
    cfg = get_config()
    dl, repo = cfg.download, cfg.noise.hf_repo
    root = noise_root()
    files = [
        item
        for sub in cfg.noise.subdirs
        for item in _list_tree(repo, f"noise/{sub}", dl.timeout_s)
        if item["type"] == "file"
    ]
    log.info("MUSAN/noise: %d files", len(files))
    jobs = [
        DownloadJob(
            url=HF_FILE.format(repo=repo, path=item["path"]),
            dest=root / item["path"],
            # Files stored in git-lfs carry a SHA-256; small plain-git files carry none.
            sha256=item.get("lfs", {}).get("oid"),
        )
        for item in files
    ]
    download_many(jobs, dl.workers, dl.retries, dl.timeout_s, desc="MUSAN/noise")
    return root


def split_pool(name: str, test_share_pct: int) -> str:
    """'train' or 'test' for a noise file, decided by a stable hash of its filename."""
    return "test" if zlib.crc32(name.encode()) % 100 < test_share_pct else "train"


@lru_cache(maxsize=1)
def noise_files() -> dict[str, list[Path]]:
    """Deterministic split of the noise files into 'train' and 'test' (by filename hash)."""
    cfg = get_config()
    root = noise_root() / "noise"
    paths = sorted(root.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no noise files in {root}; run `download` first")
    split: dict[str, list[Path]] = {pool: [] for pool in POOLS}
    for p in paths:
        split[split_pool(p.name, cfg.noise.test_share_pct)].append(p)
    return split


@lru_cache(maxsize=512)
def _load_noise(path: Path, sr: int) -> np.ndarray:
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def rng_for(clip_id: str, salt: str) -> np.random.Generator:
    """A random generator determined by (global seed, salt, clip): reproducible per clip."""
    digest = hashlib.sha256(f"{get_config().seed}:{salt}:{clip_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def noise_segment(n: int, clip_id: str, pool: str, sr: int | None = None) -> np.ndarray:
    """A noise segment of length n from the 'train' or 'test' pool, determined by clip_id."""
    sr = sr or get_config().audio.sample_rate
    files = noise_files()[pool]
    rng = rng_for(clip_id, f"noise-{pool}")
    for _ in range(20):
        x = _load_noise(files[rng.integers(len(files))], sr)
        if np.mean(x**2) > 1e-10:  # skip files that are silent
            break
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = rng.integers(0, len(x) - n + 1)
    return x[start : start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale the noise so that speech power / noise power = 10^(SNR/10), then add it."""
    p_s = np.mean(speech.astype(np.float64) ** 2)
    p_n = np.mean(noise.astype(np.float64) ** 2)
    if p_s == 0 or p_n == 0:
        return speech.copy()
    scale = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    out = speech + scale * noise
    peak = np.max(np.abs(out))
    if peak > 1.0:  # avoid clipping; scaling both parts together leaves the SNR unchanged
        out = out / peak
    return out.astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download the MUSAN noise subset.")
    ap.add_argument("command", choices=["download"])
    ap.parse_args()
    download()


if __name__ == "__main__":
    main()
