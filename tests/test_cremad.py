from __future__ import annotations

import pytest

from speech_affect.config import get_config
from speech_affect.data.cremad import parse_clip_name, parse_lfs_pointer

# ---------------------------------------------------------------------------
# Unit tests (no data needed)
# ---------------------------------------------------------------------------


def test_parse_clip_name():
    lab = parse_clip_name("1001_DFA_ANG_XX.wav")
    assert (lab.clip_id, lab.speaker_id, lab.sentence, lab.emotion, lab.intensity) == (
        "1001_DFA_ANG_XX",
        1001,
        "DFA",
        "ANG",
        "XX",
    )


@pytest.mark.parametrize("bad", ["1001_DFA_ANG.wav", "abcd_DFA_ANG_XX.wav", "x.wav"])
def test_parse_clip_name_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_clip_name(bad)


def test_parse_lfs_pointer():
    text = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:c6f94b2592487c7556bd86df7df6e8f4420d1b9c11414aeec9d8e5e640b51495\n"
        "size 72862\n"
    )
    sha, size = parse_lfs_pointer(text)
    assert sha.startswith("c6f94b25") and len(sha) == 64
    assert size == 72862


def test_parse_lfs_pointer_rejects_real_audio_header():
    with pytest.raises(ValueError):
        parse_lfs_pointer("RIFF....WAVEfmt ")


# ---------------------------------------------------------------------------
# Data integrity (needs the manifest)
# ---------------------------------------------------------------------------

cfg = get_config()


@pytest.fixture(scope="module")
def manifest():
    import pandas as pd

    path = cfg.cremad_manifest_path
    if not path.exists():
        pytest.skip("manifest not built yet")
    return pd.read_parquet(path)


@pytest.mark.data
def test_counts_match_official_description(manifest):
    c = cfg.cremad
    assert len(manifest) == c.n_clips
    assert manifest["speaker_id"].nunique() == c.n_speakers
    assert manifest["sentence"].nunique() == c.n_sentences
    assert set(manifest["emotion"]) == set(c.emotions)
    assert set(manifest["intensity"]) <= set(c.intensities)


@pytest.mark.data
def test_sex_split_matches_official_description(manifest):
    per_speaker = manifest.groupby("speaker_id")["sex"].agg(["nunique", "first"])
    assert (per_speaker["nunique"] == 1).all(), "each speaker must have a single sex"
    counts = per_speaker["first"].value_counts()
    assert counts["Male"] == cfg.cremad.n_male
    assert counts["Female"] == cfg.cremad.n_female


@pytest.mark.data
def test_hashes_match_lfs_pointers(manifest):
    assert (manifest["sha256"] == manifest["lfs_sha256"]).all()


@pytest.mark.data
def test_audio_format(manifest):
    assert (manifest["sample_rate"] == cfg.audio.sample_rate).all()
    assert (manifest["channels"] == 1).all()


@pytest.mark.data
def test_every_clip_has_listener_votes(manifest):
    assert manifest["voice_vote"].notna().all()


@pytest.mark.data
def test_invalid_clips_are_few_and_explicit(manifest):
    # Corrupt clips are excluded explicitly; if there are suddenly many, the data is broken.
    assert (~manifest["is_valid"]).sum() <= 5
