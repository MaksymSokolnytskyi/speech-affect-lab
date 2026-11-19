from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from speech_affect.channel import conditions, noise
from speech_affect.channel.noise import mix_at_snr
from speech_affect.channel.telephone import mu_law_decode, mu_law_encode, telephone

SR = 16_000


def _tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def test_telephone_keeps_length_and_dtype():
    x = _tone(1000)
    y = telephone(x)
    assert y.shape == x.shape and y.dtype == np.float32


def test_telephone_passes_voice_band():
    x = _tone(1000)
    assert _power(telephone(x)) == pytest.approx(_power(x), rel=0.1)


@pytest.mark.parametrize("freq", [5000, 7000])
def test_telephone_removes_high_frequencies(freq):
    # Nothing above 4 kHz survives 8 kHz sampling; above 3.4 kHz the band-pass cuts it too.
    assert _power(telephone(_tone(freq))) < 1e-3 * _power(_tone(freq))


def test_telephone_attenuates_low_hum():
    assert _power(telephone(_tone(60))) < 0.05 * _power(_tone(60))


def test_mu_law_roundtrip_relative_error_is_small():
    x = np.array([0.001, 0.01, 0.1, 0.5, -0.3, -0.02])
    y = mu_law_decode(mu_law_encode(x))
    # The error is relative: within a few percent for both quiet and loud samples.
    assert np.all(np.abs(y - x) / np.abs(x) < 0.1)


def test_mu_law_uses_8_bits():
    q = mu_law_encode(np.linspace(-1, 1, 10_000))
    assert q.min() == 0 and q.max() == 255


@pytest.mark.parametrize("snr", [20, 10, 5, 0])
def test_mix_at_snr_hits_target(snr):
    rng = np.random.default_rng(0)
    speech = _tone(440, amp=0.1)
    noise_ = rng.normal(0, 0.05, len(speech)).astype(np.float32)
    mixed = mix_at_snr(speech, noise_, snr)
    added = mixed - speech
    measured = 10 * np.log10(_power(speech) / _power(added))
    assert measured == pytest.approx(snr, abs=0.05)


def test_mix_at_snr_renormalises_peak_without_changing_snr():
    rng = np.random.default_rng(0)
    speech = _tone(440, amp=0.9)
    noise_ = rng.normal(0, 0.3, len(speech)).astype(np.float32)
    snr = 0.0
    mixed = mix_at_snr(speech, noise_, snr)
    peak = np.max(np.abs(mixed))
    assert peak == pytest.approx(1.0, abs=1e-6)
    # The whole mix is divided by its peak g, so speech and noise shrink together.
    unscaled = speech.astype(np.float64) + np.sqrt(
        _power(speech) / (_power(noise_) * 10 ** (snr / 10))
    ) * noise_.astype(np.float64)
    g = np.max(np.abs(unscaled))
    assert g > 1.0, "the test must exercise the renormalisation branch"
    speech_part = speech / g
    measured = 10 * np.log10(_power(speech_part) / _power(mixed - speech_part))
    assert measured == pytest.approx(snr, abs=0.05)


# ---------------------------------------------------------------------------
# Noise pools and per-clip determinism, on a small fake noise corpus
# ---------------------------------------------------------------------------

N_FAKE_FILES = 40


@pytest.fixture
def fake_noise(tmp_path, monkeypatch):
    """Noise files whose samples are all equal to a file-specific constant, so a segment
    reveals which file it was cut from."""
    d = tmp_path / "noise" / "free-sound"
    d.mkdir(parents=True)
    level = {}
    for i in range(N_FAKE_FILES):
        name = f"noise-{i:04d}.wav"
        level[name] = round(0.01 * (i + 1), 4)
        sf.write(d / name, np.full(SR // 2, level[name], dtype=np.float32), SR, subtype="FLOAT")
    monkeypatch.setattr(noise, "noise_root", lambda: tmp_path)
    noise.noise_files.cache_clear()
    yield level
    noise.noise_files.cache_clear()


def test_noise_pools_are_disjoint_and_cover_all_files(fake_noise):
    pools = noise.noise_files()
    train = {p.name for p in pools["train"]}
    test = {p.name for p in pools["test"]}
    assert train and test
    assert not train & test
    assert train | test == set(fake_noise)


def test_noise_segments_come_only_from_their_pool(fake_noise):
    pools = noise.noise_files()
    level_to_pool = {fake_noise[p.name]: pool for pool, paths in pools.items() for p in paths}
    for i in range(30):
        for pool in ("train", "test"):
            seg = noise.noise_segment(1000, f"clip{i}", pool, sr=SR)
            assert level_to_pool[round(float(seg[0]), 4)] == pool


def test_noise_segment_is_deterministic_per_clip(fake_noise):
    a = noise.noise_segment(3000, "1001_DFA_ANG_XX", "test", sr=SR)
    b = noise.noise_segment(3000, "1001_DFA_ANG_XX", "test", sr=SR)
    np.testing.assert_array_equal(a, b)
    others = [noise.noise_segment(3000, f"clip{i}", "test", sr=SR) for i in range(10)]
    assert any(not np.array_equal(a, o) for o in others)


def test_aug_train_is_deterministic_per_clip(fake_noise):
    wave = _tone(300, seconds=0.5, amp=0.3)
    for clip in ("1001_DFA_ANG_XX", "1002_IEO_SAD_HI", "1091_TSI_NEU_XX"):
        np.testing.assert_array_equal(
            conditions.aug_train(wave, clip), conditions.aug_train(wave, clip)
        )
    outputs = [conditions.aug_train(wave, f"clip{i}") for i in range(20)]
    distinct = {o.tobytes() for o in outputs}
    assert len(distinct) > 1, "augmentation should vary across clips"
