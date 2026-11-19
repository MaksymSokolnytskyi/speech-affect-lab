"""One directory per experiment run: `outputs/<experiment>/<timestamp>/` + meta.json.

meta.json records the git commit (and whether there were uncommitted changes), the config
and the time, so every number in the README can be traced back to the code that produced it.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from speech_affect.config import get_config


def _git(*args: str) -> str:
    try:
        root = get_config().paths.root
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def new_run(experiment: str, extra: dict | None = None) -> Path:
    cfg = get_config()
    run_dir = cfg.paths.outputs / experiment / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "experiment": experiment,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config": json.loads(cfg.model_dump_json()),
        **(extra or {}),
    }
    write_json(run_dir / "meta.json", meta)
    return run_dir


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def latest_run(experiment: str, marker: str = "summary.json") -> Path | None:
    """Most recent finished run of `experiment` (a run counts once `marker` is written)."""
    runs = sorted((get_config().paths.outputs / experiment).glob(f"*/{marker}"))
    return runs[-1].parent if runs else None
