"""Classic acoustic features eGeMAPSv02 (openSMILE): 88 numbers per clip.

eGeMAPS (Eyben et al., 2016) is a minimal, expert-curated standard feature set for affective
computing: pitch (F0 in semitones), loudness, voice quality (jitter, shimmer, HNR), formants,
spectral slope, and rate (loudness peaks and voiced segments per second). Each feature is a
statistic (mean, spread, percentiles) of a contour over the clip.

This is the hand-crafted baseline: if WavLM does not beat it, the large model is not needed.

Output: `data/cache/egemaps/clean.parquet`, `clip_id` + 88 columns in manifest order.
Needs the optional extra: `uv sync --extra egemaps`.

    uv run python -m speech_affect.features.egemaps
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from speech_affect.config import get_config
from speech_affect.data.cremad import load_manifest

log = logging.getLogger(__name__)


def egemaps_path() -> Path:
    return get_config().paths.cache / "egemaps" / "clean.parquet"


def smile(level: str = "functionals", num_workers: int = 1):
    """openSMILE extractor for eGeMAPSv02: "functionals" (88 per clip) or "lld" (per frame)."""
    try:
        import opensmile
    except ImportError as e:
        raise ImportError(
            "eGeMAPS needs openSMILE: install the extra with `uv sync --extra egemaps`"
        ) from e

    lvl = {
        "functionals": opensmile.FeatureLevel.Functionals,
        "lld": opensmile.FeatureLevel.LowLevelDescriptors,
    }[level]
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=lvl, num_workers=num_workers
    )


def extract(overwrite: bool = False, num_workers: int = 6) -> Path:
    out = egemaps_path()
    if out.exists() and not overwrite:
        log.info("already exists: %s", out)
        return out
    cfg = get_config()
    man = load_manifest(cfg)
    sm = smile("functionals", num_workers=num_workers)
    t0 = time.time()
    files = [str(cfg.paths.root / p) for p in man["path"]]
    feats = sm.process_files(files).reset_index(drop=True)
    if len(feats) != len(man):
        raise RuntimeError(f"openSMILE returned {len(feats)} rows instead of {len(man)}")
    feats.insert(0, "clip_id", man["clip_id"].to_numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    log.info("saved %s: %s features, %.0f s", out, feats.shape[1] - 1, time.time() - t0)
    return out


def load_egemaps() -> pd.DataFrame:
    return pd.read_parquet(egemaps_path())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Extract eGeMAPSv02 functionals for all clips.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="openSMILE worker processes")
    a = ap.parse_args()
    extract(a.overwrite, a.workers)


if __name__ == "__main__":
    main()
