from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from speech_affect.data.splits import inner_splits, random_folds, speaker_folds
from speech_affect.evaluation.bootstrap import paired_speaker_bootstrap, speaker_bootstrap
from speech_affect.evaluation.metrics import confusion, macro_f1, recall_per_class, uar

K = 6


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n_spk, per = 30, 40
    speakers = np.repeat(np.arange(n_spk), per)
    y = rng.integers(0, K, len(speakers))
    noisy = np.where(rng.random(len(y)) < 0.6, y, rng.integers(0, K, len(y)))
    return y, noisy, speakers


def test_metrics_match_sklearn(toy):
    y, pred, _ = toy
    assert uar(y, pred, K) == pytest.approx(balanced_accuracy_score(y, pred))
    assert macro_f1(y, pred, K) == pytest.approx(f1_score(y, pred, average="macro"))


def test_recall_per_class_matches_sklearn_and_marks_absent_classes(toy):
    from sklearn.metrics import recall_score

    y, pred, _ = toy
    np.testing.assert_allclose(
        recall_per_class(confusion(y, pred, K)), recall_score(y, pred, average=None)
    )
    # A class with no clips has undefined recall (NaN), not zero.
    r = recall_per_class(confusion(np.array([0, 0, 1]), np.array([0, 1, 1]), 3))
    np.testing.assert_allclose(r[:2], [0.5, 1.0])
    assert np.isnan(r[2])


def test_bootstrap_interval_contains_point(toy):
    y, pred, spk = toy
    iv, samples = speaker_bootstrap(y, pred, spk, K, n_boot=500, seed=1)
    assert iv.lo <= iv.point <= iv.hi
    assert samples.shape == (500,)
    assert iv.point == pytest.approx(balanced_accuracy_score(y, pred))


def test_bootstrap_is_deterministic(toy):
    y, pred, spk = toy
    a, _ = speaker_bootstrap(y, pred, spk, K, n_boot=200, seed=7)
    b, _ = speaker_bootstrap(y, pred, spk, K, n_boot=200, seed=7)
    assert a == b


def test_paired_difference_of_identical_systems_is_zero(toy):
    y, pred, spk = toy
    iv = paired_speaker_bootstrap(y, pred, pred, spk, K, n_boot=200, seed=1)
    assert iv.point == iv.lo == iv.hi == 0.0


def test_paired_difference_detects_better_system(toy):
    y, pred, spk = toy
    iv = paired_speaker_bootstrap(y, y, pred, spk, K, n_boot=300, seed=1)
    assert iv.lo > 0


def test_speaker_folds_do_not_leak():
    rng = np.random.default_rng(0)
    speakers = np.repeat(np.arange(40), 25)
    sex = np.repeat(rng.integers(0, 2, 40), 25)
    folds = speaker_folds(speakers, sex, n_folds=5, seed=0)
    assert set(folds) == set(range(5))
    for s in np.unique(speakers):
        assert len(set(folds[speakers == s])) == 1, "speaker in several folds"


def test_random_folds_mix_speakers():
    speakers = np.repeat(np.arange(40), 25)
    labels = np.tile(np.arange(5), 200)
    folds = random_folds(labels, n_folds=5, seed=0)
    assert any(len(set(folds[speakers == s])) > 1 for s in np.unique(speakers))


def test_inner_grouped_splits_do_not_leak():
    speakers = np.repeat(np.arange(12), 10)
    labels = np.tile(np.arange(5), 24)
    for tr, va in inner_splits(labels, speakers, 3, seed=0, grouped=True):
        assert not set(speakers[tr]) & set(speakers[va])
