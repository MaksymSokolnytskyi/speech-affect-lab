"""Hash-verified downloads.

Every file is written to `<dest>.part`, checked against its expected SHA-256, and only then
atomically renamed into place, so a crash or a corrupted transfer never leaves a file that
looks complete.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, sha256: str | None, size: int | None) -> bool:
    if not path.exists():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    return sha256 is None or sha256_of(path) == sha256


def download_verified(
    url: str,
    dest: Path,
    sha256: str | None,
    retries: int,
    size: int | None = None,
    timeout: float = 120.0,
) -> str:
    """Download `url` to `dest` unless a valid copy is already there.

    Returns "cached" or "downloaded". `sha256=None` skips the hash check (only for sources
    that publish no hash). Retries with exponential backoff on network errors and hash
    mismatches; raises `RuntimeError` after the last attempt.
    """
    if _is_valid(dest, sha256, size):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    try:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
                    f.write(r.read())
                if sha256 is not None and sha256_of(tmp) != sha256:
                    raise OSError("SHA-256 mismatch")
                tmp.replace(dest)
                return "downloaded"
            except Exception as e:  # network or hash error: try again
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
    finally:
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{url}: {last_err}")


@dataclass(frozen=True)
class DownloadJob:
    url: str
    dest: Path
    sha256: str | None
    size: int | None = None


def download_many(
    jobs: list[DownloadJob], workers: int, retries: int, timeout: float, desc: str
) -> dict[str, int]:
    """Run `download_verified` over `jobs` in a thread pool; returns counts per status.

    Raises `RuntimeError` listing (up to 20) failed files once all jobs have finished.
    """
    stats: dict[str, int] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(download_verified, j.url, j.dest, j.sha256, retries, j.size, timeout)
            for j in jobs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                status = fut.result()
                stats[status] = stats.get(status, 0) + 1
            except RuntimeError as e:
                failed.append(str(e))
    log.info("%s: %s, failed: %d", desc, stats, len(failed))
    if failed:
        raise RuntimeError("failed to download:\n" + "\n".join(failed[:20]))
    return stats
