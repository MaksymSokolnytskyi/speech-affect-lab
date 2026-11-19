"""Hash-verified download helper, exercised against local file:// URLs."""

from __future__ import annotations

import hashlib

import pytest

from speech_affect.io import DownloadJob, download_many, download_verified


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src" / "clip.wav"
    src.parent.mkdir()
    src.write_bytes(b"RIFF fake audio payload")
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def test_download_then_cached(source, tmp_path):
    src, sha = source
    dest = tmp_path / "out" / "clip.wav"
    size = src.stat().st_size
    assert download_verified(src.as_uri(), dest, sha, retries=1, size=size) == "downloaded"
    assert dest.read_bytes() == src.read_bytes()
    assert download_verified(src.as_uri(), dest, sha, retries=1, size=size) == "cached"


def test_hash_mismatch_fails_and_leaves_nothing(source, tmp_path):
    src, _ = source
    dest = tmp_path / "out" / "clip.wav"
    with pytest.raises(RuntimeError, match="SHA-256"):
        download_verified(src.as_uri(), dest, "0" * 64, retries=1)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part"))


def test_corrupt_existing_file_is_replaced(source, tmp_path):
    src, sha = source
    dest = tmp_path / "out" / "clip.wav"
    dest.parent.mkdir()
    dest.write_bytes(b"truncated")
    assert download_verified(src.as_uri(), dest, sha, retries=1) == "downloaded"
    assert dest.read_bytes() == src.read_bytes()


def test_download_many_reports_failures(source, tmp_path):
    src, sha = source
    jobs = [
        DownloadJob(src.as_uri(), tmp_path / "out" / "ok.wav", sha),
        DownloadJob(src.as_uri(), tmp_path / "out" / "bad.wav", "f" * 64),
    ]
    with pytest.raises(RuntimeError, match="failed to download"):
        download_many(jobs, workers=2, retries=1, timeout=5, desc="test")
    assert (tmp_path / "out" / "ok.wav").exists()
