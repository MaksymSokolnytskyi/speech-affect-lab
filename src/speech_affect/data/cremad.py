"""CREMA-D: audio download and manifest construction.

How it works:
1. `git clone` with `GIT_LFS_SKIP_SMUDGE=1` checks out small git-lfs pointer files instead of
   the audio. Each pointer holds the official SHA-256 and size of its file.
2. `download` fetches every WAV directly from media.githubusercontent.com and checks its
   SHA-256 against the pointer. A file that does not match is deleted and fetched again.
3. `manifest` reads every WAV, parses its filename into labels, joins actor demographics and
   the audio-only listener votes, and writes a single Parquet table: the source of truth
   about the data.

Usage:
    uv run python -m speech_affect.data.cremad download
    uv run python -m speech_affect.data.cremad manifest
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from speech_affect.config import Config, get_config
from speech_affect.io import DownloadJob, download_many, sha256_of

log = logging.getLogger(__name__)

# Listener labels in summaryTable.csv -> emotion codes used in the filenames.
VOTE_TO_EMOTION = {"A": "ANG", "D": "DIS", "F": "FEA", "H": "HAP", "N": "NEU", "S": "SAD"}


@dataclass(frozen=True)
class LfsPointer:
    rel_path: str
    sha256: str
    size: int


def parse_lfs_pointer(text: str) -> tuple[str, int]:
    """Return (sha256, size) from the text of a git-lfs pointer file."""
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines() if " " in line)
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:") or "size" not in fields:
        raise ValueError("not a git-lfs pointer")
    return oid.removeprefix("sha256:"), int(fields["size"])


@dataclass(frozen=True)
class ClipLabels:
    clip_id: str
    speaker_id: int
    sentence: str
    emotion: str
    intensity: str


def parse_clip_name(name: str) -> ClipLabels:
    """`1001_DFA_ANG_XX(.wav)` -> speaker, sentence, emotion, intensity."""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 4 or not parts[0].isdigit():
        raise ValueError(f"unexpected filename: {name}")
    speaker, sentence, emotion, intensity = parts
    return ClipLabels(stem, int(speaker), sentence, emotion, intensity)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def clone_pointers(cfg: Config) -> Path:
    repo = cfg.cremad_repo_dir
    if (repo / ".git").exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(
        ["git", "clone", "--depth", "1", "-q", cfg.cremad.repo_url, str(repo)],
        check=True,
        env=env,
    )
    return repo


def read_pointers(cfg: Config) -> list[LfsPointer]:
    wav_dir = cfg.cremad_repo_dir / "AudioWAV"
    pointers = []
    for p in sorted(wav_dir.glob("*.wav")):
        sha, size = parse_lfs_pointer(p.read_text(encoding="utf-8"))
        pointers.append(LfsPointer(f"AudioWAV/{p.name}", sha, size))
    return pointers


def download(cfg: Config) -> None:
    clone_pointers(cfg)
    out_dir = cfg.cremad_audio_dir
    jobs = [
        DownloadJob(
            url=cfg.cremad.media_url.format(path=p.rel_path),
            dest=out_dir / Path(p.rel_path).name,
            sha256=p.sha256,
            size=p.size,
        )
        for p in read_pointers(cfg)
    ]
    dl = cfg.download
    download_many(jobs, dl.workers, dl.retries, dl.timeout_s, desc="CREMA-D")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _listener_votes(cfg: Config) -> pd.DataFrame:
    """Audio-only listener votes (VoiceVote) from summaryTable.csv.

    A tie between emotions is recorded as e.g. "N:H"; `voice_vote_emotion` is then empty.
    """
    path = cfg.cremad_repo_dir / "processedResults" / "summaryTable.csv"
    df = pd.read_csv(path, usecols=["FileName", "VoiceVote"])
    df = df.rename(columns={"FileName": "clip_id", "VoiceVote": "voice_vote"})
    df["voice_vote_emotion"] = df["voice_vote"].map(VOTE_TO_EMOTION)
    return df


def _demographics(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.cremad_repo_dir / "VideoDemographics.csv")
    return df.rename(
        columns={
            "ActorID": "speaker_id",
            "Age": "age",
            "Sex": "sex",
            "Race": "race",
            "Ethnicity": "ethnicity",
        }
    )


def build_manifest(cfg: Config) -> pd.DataFrame:
    pointers = {Path(p.rel_path).name: p for p in read_pointers(cfg)}
    rows = []
    for name, ptr in tqdm(sorted(pointers.items()), desc="manifest"):
        path = cfg.cremad_audio_dir / name
        labels = parse_clip_name(name)
        info = sf.info(path)
        rows.append(
            {
                "clip_id": labels.clip_id,
                "path": path.relative_to(cfg.paths.root).as_posix(),
                "speaker_id": labels.speaker_id,
                "sentence": labels.sentence,
                "emotion": labels.emotion,
                "intensity": labels.intensity,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "n_samples": info.frames,
                "duration_s": info.frames / info.samplerate,
                "sha256": sha256_of(path),
                "lfs_sha256": ptr.sha256,
            }
        )
    df = pd.DataFrame(rows)
    df = df.merge(_demographics(cfg), on="speaker_id", how="left", validate="many_to_one")
    df = df.merge(_listener_votes(cfg), on="clip_id", how="left", validate="one_to_one")
    df["is_valid"] = df["duration_s"] >= cfg.cremad.min_duration_s
    return df


def write_manifest(cfg: Config) -> Path:
    df = build_manifest(cfg)
    out = cfg.cremad_manifest_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    bad = df.loc[~df["is_valid"], ["clip_id", "duration_s"]]
    log.info("manifest: %d clips, %d speakers -> %s", len(df), df["speaker_id"].nunique(), out)
    if len(bad):
        log.warning(
            "clips shorter than %.1f s (excluded):\n%s",
            cfg.cremad.min_duration_s,
            bad.to_string(index=False),
        )
    return out


def load_manifest(cfg: Config | None = None, valid_only: bool = True) -> pd.DataFrame:
    cfg = cfg or get_config()
    df = pd.read_parquet(cfg.cremad_manifest_path)
    return df[df["is_valid"]].reset_index(drop=True) if valid_only else df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "manifest"])
    args = parser.parse_args()
    cfg = get_config()
    if args.command == "download":
        download(cfg)
    else:
        write_manifest(cfg)


if __name__ == "__main__":
    main()
